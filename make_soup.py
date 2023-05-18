import os
import operator
from pathlib import Path

import torch
import timm
from torchvision import transforms
from timm.data import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.dataset import SDDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda:1")
models_dir = Path("./_soup_weights/sw0")
BATCH_SIZE = 1


def get_model_from_sd(state_dict, model):
    model.load_state_dict(state_dict)
    model = model.to(device)
    return model


def eval_model(net, data_loader):
    eval_steps = len(data_loader)
    with torch.no_grad():
        result_score = 0
        pbar = tqdm(data_loader)
        for bid, (images, prompts, targets) in enumerate(pbar):
            with torch.autocast(device.type):
                images, prompts = images.to(device), prompts.to(device)
                targets = targets.to(device)
                output = net(images)
            score = torch.nn.functional.cosine_similarity(output, prompts).mean()
            result_score += score.item()
            pbar.set_description(f"Eval Score: {(result_score / (bid + 1)):.6f}")
        result_score /= eval_steps
    return result_score


def gready(model_weights, dataloader, net):
    print("Greedy soup creating...")
    # get val accuracies for models
    print("Count models scores")
    models_scores = []
    for w in model_weights:
        print(w.name)
        params = torch.load(w, map_location="cpu")["net_state"]
        params = params["net_state"] if "net_state" in params else params
        net = get_model_from_sd(params, net)
        score = eval_model(net, dataloader)
        models_scores.append(score)

    # sort models by accuracy
    sorted_models_weights = sorted(
        zip(model_weights, models_scores), key=operator.itemgetter(1), reverse=True
    )
    print([[m.name, round(s, 2)] for m, s in sorted_models_weights])

    # init soup
    greedy_soup_ingredients = [sorted_models_weights[0][0].name]
    greedy_soup_params = torch.load(sorted_models_weights[0][0], map_location="cpu")[
        "net_state"
    ]
    best_val_score = sorted_models_weights[0][1]

    # Iter through models and adding them to the greedy soup.
    for i, (w, score) in enumerate(sorted_models_weights[1:], 1):
        print(f"\nModel: {w.name}")

        # Get the potential greedy soup, which consists of the greedy soup with the new model added.
        new_ingredient_params = torch.load(w, map_location="cpu")
        new_ingredient_params = (
            new_ingredient_params["net_state"]
            if "net_state" in new_ingredient_params
            else new_ingredient_params
        )

        num_ingredients = len(greedy_soup_ingredients)
        potential_greedy_soup_params = {
            k: greedy_soup_params[k].clone()
            * (num_ingredients / (num_ingredients + 1.0))
            + new_ingredient_params[k].clone() * (1.0 / (num_ingredients + 1))
            for k in new_ingredient_params
        }

        # Run the potential greedy soup on the held-out val set.
        net = get_model_from_sd(potential_greedy_soup_params, net)
        val_score = eval_model(net, dataloader)

        # If accuracy on the held-out val set increases, add the new model to the greedy soup.
        print(
            f"Potential greedy soup val score {val_score:.4f}, best: {best_val_score:.4f}."
        )
        if val_score > best_val_score:
            greedy_soup_ingredients.append(w.name)
            best_val_score = val_score
            greedy_soup_params = potential_greedy_soup_params
            print(f"Adding to soup. New soup is {greedy_soup_ingredients}")

    print(f"Final soup: {greedy_soup_ingredients}")
    torch.save(
        {"net_state": greedy_soup_params}, model_weights[0].parent / "greedy_soup.pth"
    )


def uniform(model_weights, dataloader, net, alphas=None):
    print("Uniform soup creating...")

    print("Count models scores")
    models_scores = []
    for w in model_weights:
        print(f"\n{w.name}")
        params = torch.load(w, map_location="cpu")
        params = params["net_state"] if "net_state" in params else params
        net = get_model_from_sd(params, net)
        score = eval_model(net, dataloader)
        models_scores.append(score)

    # sort models by accuracy
    sorted_models_weights = sorted(
        zip(model_weights, models_scores), key=operator.itemgetter(1), reverse=True
    )
    print([[m.name, round(s, 2)] for m, s in sorted_models_weights])

    soup_params = None
    if not alphas:
        alphas = [1] * len(model_weights)
        num_ingredients = len(sorted_models_weights)
    else:
        num_ingredients = sum(alphas)

    for i, (w, score) in enumerate(sorted_models_weights):
        print(f"\n{w.name}")
        new_ingredient_params = torch.load(w, map_location="cpu")
        new_ingredient_params = (
            new_ingredient_params["net_state"]
            if "net_state" in new_ingredient_params
            else new_ingredient_params
        )
        if not soup_params:
            soup_params = {
                k: v * (1.0 * alphas[i] / num_ingredients)
                for k, v in new_ingredient_params.items()
            }
        else:
            soup_params = {
                k: v * (1.0 * alphas[i] / num_ingredients) + soup_params[k]
                for k, v in new_ingredient_params.items()
            }

    net = get_model_from_sd(soup_params, net)
    val_score = eval_model(net, dataloader)
    print(f"Soup val score: {val_score:.4f}")
    torch.save({"net_state": soup_params}, model_weights[0].parent / "uniform_soup.pth")


def main():
    model_weights = [m for m in list(models_dir.iterdir()) if not "soup" in m.name]
    net = timm.create_model(
        "vit_large_patch14_224_clip_laion2b", False, num_classes=384
    )
    net.eval()

    dataset = SDDataset(
        Path("data/test_data"),
        Path("data/test_data/metadata.csv"),
        "test",
        transform=transforms.Compose(
            [
                transforms.Resize(size=[224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD),
            ]
        ),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        drop_last=True,
    )

    # gready(model_weights, dataloader, net)
    uniform(model_weights, dataloader, net, alphas=None)


if __name__ == "__main__":
    main()
