python scripts/data/split_data.py \
    --input_file datasets/test/mediqa-eval-2026-test-inputonly.csv \
    --PATH_JSON_DERMA "datasets/original_datasets/iiyi/test_ht_spanishtestsetcorrected.json" \
    --PATH_IMG_DERMA_ROOT "/workspace/xinzhe/MEDIQA-MEDGEMMA/images/iiyi_images_tests" \
    --PATH_JSON_WOUND "datasets/original_datasets/woundcare/test.json" \
    --PATH_IMG_WOUND_ROOT "/workspace/xinzhe/MEDIQA-MEDGEMMA/images/images_test_woundcare" \
    --output_aligned_file "datasets/test/mediqa-eval-2026-test-aligned.csv" \
    --output_folded_file "datasets/test/mediqa-eval-2026-test-folded.csv"
