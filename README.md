# EvoMega  
**Prediction of Bacteriophage Transcription Factor Binding Sites**  

## Database Download  
To download the required database, run the following command:  
```bash  
wget --save-cookies /tmp/cookies.txt --no-check-certificate "https://drive.usercontent.google.com/download?id=1569KsNmwhVuVNduQNfQ2_KLWx1v_fqGo&export=download&authuser=0&confirm=t&uuid=f30f18ad-3133-4dc9-bed3-cd95a448f69f&at=APvzH3rXKE6IjUztvq4HwPbot34Y:1734682290410" -O exclude_GPD_find_key_motif.zip && rm -f /tmp/cookies.txt  
```  

## Usage  
To use the motif analyzer, you can run the following command:  
```bash  
python scripts/motif_analyzer.py -i INPUT_FILE --input_file INPUT_FILE -o OUTPUT_PATH --output_path OUTPUT_PATH  
```  
### Example:  
```bash  
python scripts/motif_analyzer.py -i NC_002371.gbk -o scripts/output  
```  
