import torch
import sentencepiece as spm
from seq2seq.models import Seq2SeqModel


# ------------------------------
# Greedy Decode (beam=1)
# ------------------------------
def decode(model: Seq2SeqModel, src_tokens: torch.Tensor, src_pad_mask: torch.Tensor,
           max_out_len: int, tgt_tokenizer: spm.SentencePieceProcessor,
           args, device: torch.device):

    batch_size = src_tokens.size(0)
    BOS = tgt_tokenizer.bos_id()
    EOS = tgt_tokenizer.eos_id()
    PAD = tgt_tokenizer.pad_id()

    # generated token sequences
    generated = torch.full((batch_size, 1), BOS, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    # small optimization: don't move output to device every step
    src_tokens = src_tokens.to(device)
    src_pad_mask = src_pad_mask.to(device)

    for _ in range(max_out_len):

        # pad mask for decoder
        trg_pad_mask = (generated == PAD).unsqueeze(1).unsqueeze(2)

        # forward
        output = model(src_tokens, src_pad_mask, generated, trg_pad_mask)

        # greedy selection
        next_token_logits = output[:, -1, :]
        next_tokens = next_token_logits.argmax(dim=-1, keepdim=True)

        # append
        generated = torch.cat([generated, next_tokens], dim=1)

        # stop if EOS
        finished |= (next_tokens.squeeze(1) == EOS)
        if finished.all():
            break

    # strip BOS and cut at EOS
    predictions = []
    for seq in generated[:, 1:].tolist():
        if EOS in seq:
            seq = seq[: seq.index(EOS) + 1]
        predictions.append(seq)

    return predictions



# ------------------------------
# Beam Search Decode 
# ------------------------------
def beam_search_decode(model: Seq2SeqModel, src_tokens: torch.Tensor, src_pad_mask: torch.Tensor,
                       max_out_len: int, tgt_tokenizer: spm.SentencePieceProcessor,
                       args, device: torch.device, beam_size: int = 5, alpha: float = 0.7):

    model.eval()

    BOS = tgt_tokenizer.bos_id()
    EOS = tgt_tokenizer.eos_id()
    PAD = tgt_tokenizer.pad_id()

    src_tokens = src_tokens.to(device)
    src_pad_mask = src_pad_mask.to(device)

    # beam = list of (seq_tensor, score)
    beams = [(torch.tensor([[BOS]], device=device), 0.0)]

    for _ in range(max_out_len):
        new_beams = []

        for seq, score in beams:
            # if ended, keep as-is
            if seq[0, -1].item() == EOS:
                new_beams.append((seq, score))
                continue

            trg_pad_mask = (seq == PAD)[:, None, None, :]

            # forward
            logits = model(src_tokens, src_pad_mask, seq, trg_pad_mask)[:, -1, :]

            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            topk_log_probs, topk_ids = log_probs.topk(beam_size, dim=-1)

            # update beams
            for k in range(beam_size):
                next_token = topk_ids[:, k].unsqueeze(0)
                new_seq = torch.cat([seq, next_token], dim=1)
                new_score = score + topk_log_probs[:, k].item()
                new_beams.append((new_seq, new_score))

        # keep only top beam_size
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_size]

        # stop if all ended
        if all(seq[0, -1].item() == EOS for seq, _ in beams):
            break

    best_seq, _ = beams[0]
    return [best_seq.squeeze(0).tolist()]
