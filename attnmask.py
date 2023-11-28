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

    # Mask a subset based on masking_prob threshold
    generator = torch.rand(attention.shape[0], device=attention.device)
    token_mask[generator > masking_prob] = False

    return token_mask


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


def get_pred_ratio(pred_ratio=0.6, pred_ratio_var=0.0):
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