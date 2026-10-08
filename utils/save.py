import os
import torch


def save_topk_models_by_dice(
    model,
    epoch,
    avg_val_dice,
    log_dir,
    best_models,
    best_dice,
    early_stop_counter,
    patience,
    top_k=3,
    start_epoch=10,
    save_best_name="best_model.pth"
):
    """
    Save Top-K best models according to validation Dice.

    Args:
        model: PyTorch model
        epoch: current epoch
        avg_val_dice: validation Dice of current epoch
        log_dir: directory to save model weights
        best_models: list, stores top-k model info
        best_dice: current best Dice
        early_stop_counter: early stopping counter
        patience: early stopping patience
        top_k: number of best models to keep
        start_epoch: start saving after this epoch
        save_best_name: fixed filename for the best model

    Returns:
        best_models, best_dice, early_stop_counter
    """

    if epoch <= start_epoch:
        return best_models, best_dice, early_stop_counter

    os.makedirs(log_dir, exist_ok=True)

    should_save = False

    if len(best_models) < top_k:
        should_save = True
    else:
        min_top_dice = min(item["dice"] for item in best_models)
        if avg_val_dice > min_top_dice:
            should_save = True

    if should_save:
        save_name = f"best_epoch_{epoch:03d}_dice_{avg_val_dice:.4f}.pth"
        save_path = os.path.join(log_dir, save_name)

        torch.save(model.state_dict(), save_path)

        best_models.append({
            "dice": avg_val_dice,
            "path": save_path,
            "epoch": epoch
        })

        # 按 Dice 从高到低排序
        best_models = sorted(
            best_models,
            key=lambda x: x["dice"],
            reverse=True
        )

        # 只保留 Top-K
        while len(best_models) > top_k:
            removed = best_models.pop(-1)

            if os.path.exists(removed["path"]):
                os.remove(removed["path"])
                print(
                    f"🗑️ Remove Model | "
                    f"Epoch: {removed['epoch']} | "
                    f"Dice: {removed['dice']:.4f}"
                )

        best_dice = best_models[0]["dice"]

        # 如果当前模型是新的 Top-1，则额外保存为固定名字 best_model.pth
        if best_models[0]["path"] == save_path:
            best_model_path = os.path.join(log_dir, save_best_name)
            torch.save(model.state_dict(), best_model_path)

        early_stop_counter = 0

        print(f"✅ Save Top-{top_k} Model | Epoch: {epoch} | Dice: {avg_val_dice:.4f}")
        print("Current Top Models:")

        for rank, item in enumerate(best_models, start=1):
            print(
                f"  Top {rank}: "
                f"Epoch {item['epoch']} | "
                f"Dice {item['dice']:.4f} | "
                f"{os.path.basename(item['path'])}"
            )

    else:
        early_stop_counter += 1
        print(
            f"No Top-{top_k} improvement. "
            f"Early stopping counter: {early_stop_counter}/{patience}"
        )

    return best_models, best_dice, early_stop_counter