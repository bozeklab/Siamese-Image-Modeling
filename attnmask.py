import torch
import random


def AttMask(attention, masking_prob, masking_mode, masking_ratio, show_ratio, show_max):
    # Get AttMask (High, Hints or Low)
    masks = get_mask(attention,
                     masking_prob,
                     masking_mode,
                     masking_ratio
                     )

    # For AttMask-Hints, randomly reveal some of the most highly attended tokens
    if masking_mode == 'attmask_hint':
        # Get a mask of the top show(%) most attended tokens
        top_masks = get_mask(attention,
                             1,
                             masking_mode,
                             show_max
                             )
        # Reveal some of the most attended tokens
        masks = show_hints(top_masks, masks, show_ratio)

    return masks


def get_mask(attention, masking_prob, masking_mode, masking_ratio):
    # Token masking
    token_mask = attention_masking(attention, masking_mode, masking_ratio)
    flipped_tensor = token_mask.clone()

    # Mask a subset based on masking_prob threshold
    for row_idx in range(flipped_tensor.size(0)):
        # Get the indices where the boolean mask is True
        true_indices = torch.nonzero(token_mask[row_idx]).squeeze()

        # Randomly select a fraction p of True values to flip to False
        num_indices_to_flip = int(len(true_indices) * (1 - masking_prob))
        selected_indices = torch.randperm(len(true_indices))[:num_indices_to_flip]

        # Flip the selected True values to False
        flipped_tensor[row_idx, true_indices[selected_indices]] = False

    return flipped_tensor

def attention_masking(attention, masking_mode, masking_ratio):
    N = int(attention.shape[1] * masking_ratio)
    attn_mask = torch.zeros(attention.shape, dtype=torch.bool, device=attention.device)

    if masking_mode in ['attmask_high', 'attmask_hint']:
        idx = torch.argsort(attention, descending=True)[:, :N]
    elif masking_mode == 'attmask_low':
        idx = torch.argsort(attention, descending=False)[:, :N]
    else:
        raise ('Use attmask_high, attmask_hint or attmask_low')

    attn_mask.scatter_(1, idx, True)

    return attn_mask


def get_pred_ratio(pred_ratio=0.3, pred_ratio_var=0.2):
    if isinstance(pred_ratio, list):
        pred_ratio = []
        for prm, prv in zip(pred_ratio, pred_ratio_var):
            assert prm >= prv
            pr = random.uniform(prm - prv, prm + prv) if prv > 0 else prm
            pred_ratio.append(pr)
        pred_ratio = random.choice(pred_ratio)
    else:
        assert pred_ratio >= pred_ratio_var
        pred_ratio = random.uniform(pred_ratio - pred_ratio_var, pred_ratio + \
                                    pred_ratio_var) if pred_ratio_var > 0 else pred_ratio

    return pred_ratio


def show_hints(top_masks, masks, show_ratio):
    _, n_tokens = masks.shape
    reveal_tokens = int(show_ratio * n_tokens)

    selected_high = torch.multinomial(top_masks.float(), reveal_tokens)

    masks.scatter_(1, selected_high, False)

    return masks