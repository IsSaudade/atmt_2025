import torch
import sentencepiece as spm
from seq2seq.models import Seq2SeqModel

def decode(model: Seq2SeqModel, src_tokens: torch.Tensor, src_pad_mask: torch.Tensor, max_out_len: int,
           tgt_tokenizer: spm.SentencePieceProcessor, args, device: torch.device):
    batch_size = src_tokens.size(0)
    BOS = tgt_tokenizer.bos_id()
    EOS = tgt_tokenizer.eos_id()
    PAD = tgt_tokenizer.pad_id()

    # --- run encoder once ---
    with torch.no_grad():
        enc_out = model.encoder(src_tokens, src_pad_mask)

    # make sure it's a tuple (old models return only 1 tensor)
    if isinstance(enc_out, tuple):
        encoder_out, encoder_pad_mask = enc_out
    else:
        encoder_out, encoder_pad_mask = enc_out, None

    generated = torch.full((batch_size, 1), BOS, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for t in range(max_out_len):
        # truncate if too long
        max_len = model.decoder.pos_embed.size(1)
        if generated.size(1) > max_len:
            generated = generated[:, :max_len]

        # ensure long dtype
        generated = generated.long()

        # create padding mask
        trg_pad_mask = (generated == PAD).unsqueeze(1).unsqueeze(2)

        # --- decoder only ---
        with torch.no_grad():
            decoder_out = model.decoder(generated, trg_pad_mask, encoder_out, encoder_pad_mask)

        # last step logits
        next_token_logits = decoder_out[:, -1, :]
        next_tokens = next_token_logits.argmax(dim=-1, keepdim=True).long()

        generated = torch.cat([generated, next_tokens], dim=1)

        finished |= (next_tokens.squeeze(1) == EOS)
        if finished.all():
            break

    # post-process
    predicted_tokens = []
    for seq in generated[:, 1:].tolist():
        if EOS in seq:
            seq = seq[: seq.index(EOS) + 1]
        predicted_tokens.append(seq)
    return predicted_tokens


def beam_search_decode(model: Seq2SeqModel, src_tokens: torch.Tensor, src_pad_mask: torch.Tensor, max_out_len: int,
                       tgt_tokenizer: spm.SentencePieceProcessor, args, device: torch.device,
                       beam_size: int = 5, alpha: float = 0.7):
    model.eval()
    BOS, EOS, PAD = tgt_tokenizer.bos_id(), tgt_tokenizer.eos_id(), tgt_tokenizer.pad_id()

    # --- run encoder once ---
    with torch.no_grad():
        enc_out = model.encoder(src_tokens, src_pad_mask)

    if isinstance(enc_out, tuple):
        encoder_out, encoder_pad_mask = enc_out
    else:
        encoder_out, encoder_pad_mask = enc_out, None

    beams = [(torch.tensor([[BOS]], device=device, dtype=torch.long), 0.0)]

    for _ in range(max_out_len):
        new_beams = []
        for seq, score in beams:
            seq = seq.long()

            if seq[0, -1].item() == EOS:
                new_beams.append((seq, score))
                continue

            with torch.no_grad():
                max_len = model.decoder.pos_embed.size(1)
                if seq.size(1) > max_len:
                    seq = seq[:, :max_len]

                trg_pad_mask = (seq == PAD)[:, None, None, :]
                logits = model.decoder(seq, trg_pad_mask, encoder_out, encoder_pad_mask)[:, -1, :]

                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                topk_log_probs, topk_ids = log_probs.topk(beam_size, dim=-1)

            for k in range(beam_size):
                new_seq = torch.cat([seq, topk_ids[:, k].unsqueeze(0)], dim=1).long()
                new_score = score + topk_log_probs[:, k].item()
                new_beams.append((new_seq, new_score))

        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]

        if all(s[0, -1].item() == EOS for s, _ in beams):
            break

    best_seq, _ = beams[0]
    return [best_seq.squeeze(0).tolist()]
