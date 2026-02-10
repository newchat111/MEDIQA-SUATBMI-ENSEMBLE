from utils.eval.eval_csv import get_csv_scores
import draccus
from dataclasses import dataclass

@dataclass
class paths:
    input_path:str = ""
    output_path:str = ""
    metrics:str = ""
    
