python scripts/data/split_data.py \
    --input_file datasets/mediqa-eval-2026-valid.csv \
    --PATH_JSON_DERMA "datasets/original_datasets/iiyi/valid_ht.json" \
    --PATH_IMG_DERMA_ROOT "/data/liyuan/datasets/competition/derma/data/iiyi/images_final/images_valid" \
    --PATH_JSON_WOUND "datasets/original_datasets/woundcare/valid.json" \
    --PATH_IMG_WOUND_ROOT "/data/liyuan/datasets/competition/woundcare/dataset-challenge-mediqa-2025-wv/images_final/images_valid" \
    --output_aligned_file "my_test/complete_test/mediqa-eval-2026-valid-aligned.csv" \
    --output_folded_file "my_test/complete_test/mediqa-eval-2026-valid-folded.csv"