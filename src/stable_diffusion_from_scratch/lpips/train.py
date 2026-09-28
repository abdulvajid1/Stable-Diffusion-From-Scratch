import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from PIL import Image
import argparse
from accelarate import Accelerator


from .lpips import LPIPS, LpipsDiffToLogits

class BAPPSDataset(Dataset):
    def __init__(self,
                 path_to_root:str,
                 train=True,
                 dirs=None,
                 img_size=64):
        super().__init__()


        if train:
            split = 'train'
            if dirs is None:
                dirs = ['cnn', 'mix', 'traditional']

        else:
            split = "val"
            if dirs is None:
                dirs = ['cnn', 'color', 'deblur', 'frameinterp', 'superrs', 'traditional']


        if isinstance(dirs, str):
            dirs = [dirs]

        path_to_dirs = [os.path.join(path_to_root, split, dir) for dir in dirs]

        self.samples = self._generate_dataset(path_to_dirs)

        self.transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5,], [0.5, 0.5, 0.5])
            ]
        )


    def _generate_dataset(self, path_to_dirs):
        samples = []
        for dir in path_to_dirs:
            path_to_p0 = os.path.join(dir, "p0")
            path_to_p1 = os.path.join(dir, "p1")
            path_to_ref = os.path.join(dir, "ref")
            path_to_target = os.path.join(dir, "judge")

            file_idx = [file.split(".") for file in os.listdir(path_to_p0)]
            for idx in file_idx:
                p0 = os.path.join(path_to_p0, f"{idx}.png")
                p1 = os.path.join(path_to_p1, f"{idx}.png")
                ref = os.path.join(path_to_ref, f"{idx}.png")
                target = os.path.join(path_to_target, f"{idx}.png")
                samples.append((p0, p1, ref, target))


        return samples


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img1, img2, ref, target = self.samples[index]
        img1 = self.transform(Image.open(img1).convert("RGB"))
        img2 = self.transforms(Image.open(img2).convert("RGB"))
        ref = self.transforms(Image.open(ref).convert("RGB"))
        target = np.load(target)[0]

        return img1, img2, ref, target



class LpipsForTraining(nn.Module):
    def __init__(self,
                 pretrained_backbone=True,
                 train_backbone=False,
                 use_dropout=True,
                 img_range='minus_one_to_one',
                 middle_chennel=32):
        
        super().__init__()

        self.lpips = LPIPS(image_range=img_range, 
                           use_dropout=use_dropout)

        self.diff_to_logits = LpipsDiffToLogits(middle_channel=middle_chennel)


    def bce_rank_loss(self, diff1, diff2, target):
        output = self.diff_to_logits(diff1, diff2)
        output = output.reshape(*target.shape)
        loss = F.binary_cross_entropy_with_logits(output, target=target)
        return loss


    def clamp_weights(self):
        for module in self.lpips.modules:
            if hasattr(module, "weight") and module.kernel_size == (1, 1):
                module.weight.data = torch.clamp(module.weight.data, min=0.)

    def checkpoint_model(self, path_to_checkpoint, checkpoint_name):
        path_to_lpips = os.path.join(path_to_checkpoint, checkpoint_name)

        print(f"Saving Lpips to {path_to_checkpoint}")
        torch.save(self.lpips.state_dict(), path_to_lpips)


    def forward(self, img1, img2, ref, target):
        diff1 = self.lpips(img1, ref)
        diff2 = self.lpips(img2, ref)
        loss = self.bce_rank_loss(diff1, diff2, target)

        return loss, diff1, diff2



class LRScheduler:
    def __init__(self, optimizer, initial_lr, total_iteration, decay_iteration, min_lr=None):

        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.total_iteration = total_iteration
        self.decay_iteratoin = decay_iteration

        self.constant_iteration = total_iteration - decay_iteration
        self.min_lr = min_lr if min_lr is not None else 0

        self.current_step = 0


    def step(self):
        if self.current_step < self.constant_iteration:
            lr = self.initial_lr
        else:
            decay_ratio = (self.current_step - self.constant_iteration) / self.decay_iteratoin

            lr = max(self.min_lr, self.initial_lr * (1 - decay_ratio))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        self.current_step += 1

def compute_accuracy(diff1, diff2, target):
    preds = (diff2 < diff1).flatten().int()
    target = target.flatten()

    accuracy = torch.mean(preds * target + (1 - preds) * (1 - target))
    return accuracy


def trainer(args):


    if not os.path.exists(args.working_dir):
        os.makedirs(args.working_dir, exist_ok=True)

    training_set = BAPPSDataset(path_to_root=args.path_to_root, train=True, img_size=args.img_size)
    train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    model = LpipsForTraining(
        pretrained_backbone=args.pretrained_backbone,
        train_backbone=args.train_backbone,
        use_dropout=args.use_dropout,
        img_range=args.img_range,
        middle_channel=args.middle_channel
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    scheduler = LRScheduler(
        optimizer=optimizer, 
        initial_lr=args.initial_lr, 
        total_iteration=len(train_loader) * args.num_epochs, 
        decay_iteration=len(train_loader) * args.decay_epochs
        )

    model, optimizer, trainloader, scheduler = accelerator.prepare(
        model, optimizer, trainloader, scheduler
    )

    total_training_iterations = len(train_loader) * args.num_epochs
    accelerator.print("Training for {} Iterration".format(total_training_iterations))

    iteration = 0
    while iteration < total_training_iterations:
        for batch in trainloader:
            img1, img2, ref, target = (i.to(accelerator.device) for i in batch)
            loss, diff1, diff2 = model(img1, img2, ref, target)

            accelerator.backward(loss)
            accelerator.clip_norm_grad(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            accelerator.unwrap_model(model).clamp_weights()
            scheduler.step()
            iteration += 1

            if iteration % args.logging_steps == 0:
                accuracy = compute_accuracy(diff1, diff2, target)
                accuracy = torch.mean(accelerator.gather_for_metric(accuracy)).item()
                loss = torch.mean(accelerator.gather_for_metric(loss)).item()

            log = {
                "iteration": iteration,
                "loss": loss,
                "accuracy": accuracy,
                "lr": optimizer.param_group[0]['lr']
            }
            accelerator.print(log)



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="LPIPS Training Arguments")

    parser.add_argument("--path_to_root", 
                        help="Path to BAPPS Dataset Root",
                        required=True, 
                        type=str)
    
    parser.add_argument("--work_dir", 
                        help="Path to where you want to save checkpoints",
                        required=True, 
                        type=str)
    
    parser.add_argument("--checkpoint_name", 
                        help="Name for the final checkpoint",
                        default="lpips_vgg.pt",
                        required=False, 
                        type=str)
    
    parser.add_argument("--batch_size", 
                        help="Batch size to train with (will get multipled by n_gpus)",
                        default=64, 
                        required=False, 
                        type=int)
    
    parser.add_argument("--eval_batch_size", 
                        help="Batch size to train with",
                        default=256, 
                        required=False, 
                        type=int)
    
    parser.add_argument("--img_size", 
                        help="What image size do you want to use?",
                        default=64, 
                        required=False, 
                        type=int)

    parser.add_argument("--num_workers",
                        help="DataLoader workers", 
                        default=8, 
                        required=False,
                        type=int)
    
    parser.add_argument("--num_epochs",
                        help="How many epochs do you want to train for?", 
                        default=10, 
                        required=False,
                        type=int)
    
    parser.add_argument("--decay_epochs",
                        help="How many epochs do you want linearly decay LR?", 
                        default=5, 
                        required=False,
                        type=int)
    
    parser.add_argument("--initial_lr",
                        help="What learning rate do you want to use?", 
                        default=1e-4, 
                        required=False,
                        type=float)
    
    parser.add_argument("--logging_steps", 
                        help="After how many iterations do you want to print logs",
                        default=1000, 
                        required=False, 
                        type=int)
    
    parser.add_argument("--pretrained_backbone", 
                        help="Use a pretrained backbone", 
                        action='store_true')

    parser.add_argument("--train_backbone", 
                        help="Allow training of the backbone", 
                        action='store_true')

    parser.add_argument("--use_dropout", 
                        help="Enable dropout layers", 
                        action='store_true')

    parser.add_argument("--img_range", 
                        help="Image range options: 'minus_one_to_one' or 'zero_to_one' (default: 'minus_one_to_one')", 
                        default="minus_one_to_one", 
                        required=False,
                        type=str)

    parser.add_argument("--middle_channels", 
                        help="Number of middle channels in the model (default: 32)", 
                        default=32, 
                        required=False,
                        type=int)
    
    parser.add_argument("--evaluation_only", 
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        type=bool)
    
    parser.add_argument("--eval_lpips_pkg",
                        action=argparse.BooleanOptionalAction, 
                        default=False, 
                        type=bool)
    
    parser.add_argument("--mixed_precision",
                        action=argparse.BooleanOptionalAction, 
                        default=False, 
                        type=bool)
    
    args = parser.parse_args()

    ### Define Accelerator ###
    accelerator = Accelerator()

    if not args.evaluation_only:
        trainer(args)

    ### Evaluate on one GPU only ###
    if accelerator.is_main_process:
        eval(args)

    accelerator.wait_for_everyone()
    accelerator.end_training()
    

        