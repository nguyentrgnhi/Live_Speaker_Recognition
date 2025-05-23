import multiprocessing
import torch

import dataset
import feature_extraction
import myconfig
import neural_net

import sounddevice as sd
import numpy as np
import wave
import subprocess
import tempfile
import time


def run_inference(features, encoder, full_sequence=myconfig.USE_FULL_SEQUENCE_INFERENCE):
    """Get the embedding of an utterance using the encoder."""
    if full_sequence:
        # Full sequence inference.
        batch_input = torch.unsqueeze(torch.from_numpy(features), dim=0).float().to(myconfig.DEVICE)
        batch_output = encoder(batch_input)
        return batch_output[0, :].cpu().data.numpy()
    else:
        sliding_windows = feature_extraction.extract_sliding_windows(features)
        if not sliding_windows:
            return None
        batch_input = torch.from_numpy(np.stack(sliding_windows)).float().to(myconfig.DEVICE)
        batch_output = encoder(batch_input)

        # Aggregate the inference outputs from sliding windows.
        aggregated_output = torch.mean(batch_output, dim=0, keepdim=False).cpu()
        return aggregated_output.data.numpy()

def cosine_similarity(a, b):
    """Compute cosine similarity between two embeddings."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class TripletScoreFetcher:
    """Class for computing triplet scores with multi-processing."""

    def __init__(self, spk_to_utts, encoder, NUM_EVAL_TRIPLETS):
        self.spk_to_utts = spk_to_utts
        self.encoder = encoder
        self.num_eval_triplets = NUM_EVAL_TRIPLETS

    def __call__(self, i):
        """Get the labels and scores from a triplet."""
        anchor, pos, neg = feature_extraction.get_triplet_features(self.spk_to_utts)
        anchor_embedding = run_inference(anchor, self.encoder)
        pos_embedding = run_inference(pos, self.encoder)
        neg_embedding = run_inference(neg, self.encoder)
        if (anchor_embedding is None) or (pos_embedding is None) or (neg_embedding is None):
            # Some utterances might be smaller than a single sliding window.
            return [], []
        triplet_labels = [1, 0]
        triplet_scores = [
            cosine_similarity(anchor_embedding, pos_embedding),
            cosine_similarity(anchor_embedding, neg_embedding)]
        print("triplets evaluated:", i, "/", self.num_eval_triplets)
        return (triplet_labels, triplet_scores)

def compute_scores(encoder, spk_to_utts, num_eval_triplets=myconfig.NUM_EVAL_TRIPLETS):
    """Compute cosine similarity scores from testing data."""
    labels = []
    scores = []
    fetcher = TripletScoreFetcher(spk_to_utts, encoder, num_eval_triplets)

    # CUDA does not support multi-processing, so using a ThreadPool.
    with multiprocessing.Pool(myconfig.NUM_PROCESSES) as pool:
        while num_eval_triplets > len(labels) // 2:
            label_score_pairs = pool.map(fetcher, range(len(labels) // 2, num_eval_triplets))
            for triplet_labels, triplet_scores in label_score_pairs:
                labels += triplet_labels
                scores += triplet_scores
    print("Evaluated", len(labels) // 2, "triplets in total")

    return labels, scores

def compute_eer(labels, scores):
    """Compute the Equal Error Rate (EER)."""
    if len(labels) != len(scores):
        raise ValueError("Length of labels and scored must match")
    eer_threshold = None
    eer = None
    min_delta = 1
    threshold = 0.0
    while threshold < 1.0:
        accept = [score >= threshold for score in scores]
        fa = [a and (1-l) for a, l in zip(accept, labels)]
        fr = [(1-a) and l for a, l in zip(accept, labels)]
        far = sum(fa) / (len(labels) - sum(labels))
        frr = sum(fr) / sum(labels)
        delta = abs(far - frr)
        if delta < min_delta:
            min_delta = delta
            eer = (far + frr) / 2
            eer_threshold = threshold
        threshold += myconfig.EVAL_THRESHOLD_STEP

    return eer, eer_threshold

def run_eval():
    """Run evaluation of the saved model on test data."""
    start_time = time.time()
    spk_to_utts =  dataset.get_librispeech_speaker_to_utterance(myconfig.MY_TEST_DATA_DIR)
    print("Evaluation data:", myconfig.MY_TEST_DATA_DIR)
    encoder = neural_net.get_speaker_encoder(r"/Users/nguyennhi/PycharmProjects/pythonProject1/saved_model_20250522135210.pt")
    labels, scores = compute_scores(encoder, spk_to_utts, myconfig.NUM_EVAL_TRIPLETS)
    print(labels, scores)
    eer, eer_threshold = compute_eer(labels, scores)
    eval_time = time.time() - start_time
    print("Finished evaluation in", eval_time, "seconds")
    print("eer_threshold =", eer_threshold, "eer =", eer)


def run_predict(flac_file, encoder, ref_emb_dict):
    print(f"Selected file for prediction: {flac_file}")

    # Fix: Pass the file path, not the waveform
    print("-")
    features = feature_extraction.extract_features(flac_file)

    # Get the embedding of the input audio
    embedding = run_inference(features, encoder)
    print("-")
    if embedding is None:
        print("Error: Could not extract valid features from the input file.")
        return None

    top_matches = []  # List to store (speaker_id, score)

    for spk_id, ref_emb in ref_emb_dict.items():
        if ref_emb is None:
            continue  # Skip if invalid

        score = cosine_similarity(embedding, ref_emb)
        top_matches.append((spk_id, score))

    print(top_matches)

    # Sort by score in descending order and take the top 3
    top_matches = sorted(top_matches, key=lambda x: x[1], reverse=True)[:3]

    print(top_matches)

    print("Top 3 Predicted Speakers:")
    for rank, (spk_id, score) in enumerate(top_matches, start=1):
        print(f"{rank}. Speaker ID: {spk_id} (Similarity Score: {score:.4f})")


    #print(f"Predicted Speaker ID: {best_match} (Similarity Score: {best_score:.4f})")
    #return best_match

def record_audio(duration=3, sample_rate=16000):
    print("Recording...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.int16)
    sd.wait()  # Wait until recording is finished
    print("Recording complete.")

    # Save as WAV file (temporary)
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with wave.open(temp_wav.name, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())

    return temp_wav.name

def convert_wav_to_flac(wav_file):
    temp_flac = tempfile.NamedTemporaryFile(delete=False, suffix=".flac")
    subprocess.run(["ffmpeg", "-y", "-i", wav_file, "-ac", "1", "-ar", "16000", temp_flac.name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return temp_flac.name


if __name__ == "__main__":
    my_dir = myconfig.MY_TEST_DATA_DIR  # Ensure it's set correctly
    spk_to_utts = dataset.get_librispeech_speaker_to_utterance(my_dir)
    encoder = neural_net.get_speaker_encoder(
        r"/Users/nguyennhi/PycharmProjects/pythonProject1/saved_model_20250522135210.pt"
    )

    ref_emb_dict = {}

    print("Rebuilding Ref_Embeddings...")
    for spk_id, utt_list in spk_to_utts.items():
        for utt in utt_list:
            features_utt = feature_extraction.extract_features(utt)
            ref_embedding = run_inference(features_utt, encoder)
            count = 1
            new_id = spk_id + "-0"
            while new_id in ref_emb_dict:
                c = new_id.find("-")
                new_id = new_id[:c+1] + str(count)
                count = count + 1
            ref_emb_dict[new_id] = ref_embedding



    while True:
        wav_file = record_audio(duration=10)  # Record 3 seconds
        flac_file = convert_wav_to_flac(wav_file)  # Convert to FLAC
        predicted_speaker = run_predict(flac_file, encoder, ref_emb_dict)  # Predict
        time.sleep(1)  # Small delay before next recording


    print("----")
    flac_file = "/Users/nguyennhi/PycharmProjects/pythonProject1/test/322/2025-03-20_14-43-37/preston-2025-03-20_14-43-37-1.flac"
    predicted_speaker = run_predict(flac_file, encoder, ref_emb_dict)
    print(f"Predicted Speaker 1: {predicted_speaker}\n")

    print("----")
    exit(0)




    #

    cache_path = "ref_embeddings.pkl"



    if os.path.exists(cache_path):
        use_cache = input("Load saved Ref_Embeddings? (y/n): ").strip().lower()
        if use_cache == "y":
            with open(cache_path, "rb") as f:
                ref_emb_dict = pickle.load(f)
            print("Loaded saved Ref_Embeddings.")
        else:
            print("Rebuilding Ref_Embeddings...")
            for spk_id, utt_list in spk_to_utts.items():
                for utt in utt_list:
                    features_utt = feature_extraction.extract_features(utt)
                    ref_embedding = run_inference(features_utt, encoder)
                    ref_emb_dict[spk_id] = ref_embedding
            with open(cache_path, "wb") as f:
                pickle.dump(ref_emb_dict, f)
            print("Saved new Ref_Embeddings.")
    else:
        print("No cache found. Building Ref_Embeddings...")
        for spk_id, utt_list in spk_to_utts.items():
            for utt in utt_list:
                features_utt = feature_extraction.extract_features(utt)
                ref_embedding = run_inference(features_utt, encoder)
                ref_emb_dict[spk_id] = ref_embedding
        with open(cache_path, "wb") as f:
            pickle.dump(ref_emb_dict, f)
        print("Saved new Ref_Embeddings.")

    #
    #
    #predicted_speaker = run_predict(flac_file, encoder, ref_emb_dict)  # Predict
    #print(f"Predicted Speaker 1: {predicted_speaker}\n")
    #
    # flac_file = "/Users/nguyennhi/PycharmProjects/pythonProject1/test/2911/2025-04-02_14-27-28/hayeon-2025-04-02_14-27-28-18.flac"
    #
    # predicted_speaker = run_predict(flac_file, encoder, ref_emb_dict)  # Predict
    # print(f"Predicted Speaker 1: {predicted_speaker}\n")
    #
    #
    flac_file = "/Users/nguyennhi/PycharmProjects/pythonProject1/nhi-2025-04-02_13-45-45-1.flac"
     #
    predicted_speaker = run_predict(flac_file, encoder, ref_emb_dict)  # Predict
    print(f"Predicted Speaker 1: {predicted_speaker}\n")

    flac_file = "/Users/nguyennhi/PycharmProjects/pythonProject1/test/126/2025-04-02_14-06-00/duc-2025-04-02_14-06-00-4.flac"
     #
    predicted_speaker = run_predict(flac_file, encoder, ref_emb_dict)  # Predict
    print(f"Predicted Speaker 1: {predicted_speaker}\n")


    flac_file = "/Users/nguyennhi/PycharmProjects/pythonProject1/preston-2025-03-20_14-54-22-1.flac"
     #
    predicted_speaker = run_predict(flac_file, encoder, ref_emb_dict)  # Predict
    print(f"Predicted Speaker 1: {predicted_speaker}\n")


    exit(0)







