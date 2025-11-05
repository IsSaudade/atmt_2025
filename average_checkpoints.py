import torch
import argparse
import os
import glob
import logging

def get_args():
    parser = argparse.ArgumentParser(description="Average the last N checkpoints.")
    parser.add_argument('--checkpoint-dir', required=True, type=str, help='Directory containing checkpoints')
    parser.add_argument('--output', required=True, type=str, help='Output path for the averaged checkpoint')
    parser.add_argument('--n', type=int, default=5, help='Number of last checkpoints to average')
    return parser.parse_args()

def main(args):
    logging.basicConfig(level=logging.INFO)

    # Find checkpoint files
    checkpoint_files = glob.glob(os.path.join(args.checkpoint_dir, 'checkpoint_*.pt'))
    
    # Filter out best and last checkpoints, and sort by epoch number
    epoch_checkpoints = []
    for f in checkpoint_files:
        basename = os.path.basename(f)
        if basename.startswith('checkpoint_') and basename != 'checkpoint_best.pt' and basename != 'checkpoint_last.pt':
            try:
                epoch_num = int(basename.split('_')[1].split('.')[0])
                epoch_checkpoints.append((epoch_num, f))
            except (ValueError, IndexError):
                continue
    
    if not epoch_checkpoints:
        logging.error("No epoch checkpoints found in the specified directory.")
        return

    epoch_checkpoints.sort(key=lambda x: x[0], reverse=True)

    # Select the last N checkpoints
    checkpoints_to_average = epoch_checkpoints[:args.n]
    if len(checkpoints_to_average) < args.n:
        logging.warning(f"Found only {len(checkpoints_to_average)} checkpoints, averaging these.")
    if not checkpoints_to_average:
        logging.error("No checkpoints to average.")
        return

    logging.info(f"Averaging the following checkpoints: {[os.path.basename(f) for _, f in checkpoints_to_average]}")

    # Load the first checkpoint to initialize the averaged state dict
    main_checkpoint = torch.load(checkpoints_to_average[0][1], map_location='cpu')
    avg_state_dict = main_checkpoint['model']

    # Sum the state dicts of the other checkpoints
    for _, filepath in checkpoints_to_average[1:]:
        checkpoint = torch.load(filepath, map_location='cpu')
        for key in avg_state_dict:
            avg_state_dict[key] = avg_state_dict[key] + checkpoint['model'][key]

    # Average the state dict
    for key in avg_state_dict:
        if avg_state_dict[key].is_floating_point():
            avg_state_dict[key] = avg_state_dict[key] / len(checkpoints_to_average)
        else:
            # For non-floating point tensors, just use the value from the last checkpoint
            avg_state_dict[key] = main_checkpoint['model'][key]


    # Create the new checkpoint, using metadata from the last checkpoint
    final_checkpoint = main_checkpoint
    final_checkpoint['model'] = avg_state_dict
    final_checkpoint['epoch'] = checkpoints_to_average[0][0] # Mark it with the latest epoch

    # Save the averaged checkpoint
    torch.save(final_checkpoint, args.output)
    logging.info(f"Saved averaged checkpoint to {args.output}")

if __name__ == '__main__':
    args = get_args()
    main(args)
