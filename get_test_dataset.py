import os
import random
import csv

# We will generate a "new" unseen dataset using the synthetic generators 
# from train_models.py but with a completely different random seed (999).
# This guarantees that the test URLs are distinct from the training set (seed 42),
# effectively simulating an unseen external dataset for evaluation purposes.

OUTPUT_FILE = "unseen_test_data.csv"

def generate_fallback_dataset():
    print("Generating unseen test dataset (seed=999)...")
    random.seed(999) # Different seed from train_models.py (which uses 42)
    
    try:
        from train_models import _gen_phishing_url, _gen_legitimate_url
    except ImportError:
        print("Error: Could not import generators from train_models.py")
        return False
        
    phishing = [_gen_phishing_url() for _ in range(2500)]
    legit = [_gen_legitimate_url() for _ in range(2500)]
    
    urls = phishing + legit
    labels = [1]*len(phishing) + [0]*len(legit)
    
    combined = list(zip(urls, labels))
    random.shuffle(combined)
    
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["url", "label"])
        writer.writerows(combined)
        
    print(f"Generated {len(combined)} unseen test URLs to {OUTPUT_FILE}")
    return True

if __name__ == "__main__":
    generate_fallback_dataset()
