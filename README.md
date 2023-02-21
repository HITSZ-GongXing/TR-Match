# TR-Match
This is the code for the 2023 DASFAA paper: Temporal-Relational Matching Network for Few-shot Temporal Knowledge Graph Completion. Please cite our paper if you use the code or datasets:
```latex
@inproceedings{Gong2023TRmatch
  author    = {Xing Gong and
               Jianyang Qin and
               Heyan Chai and
               Ye Ding and
               Yan Jia and
               Qing Liao},
  title     = {Temporal-Relational Matching Network for Few-shot Temporal Knowledge Graph Completion
},
  booktitle = {Database Systems for Advanced Applications - 28th International Conference,
               {DASFAA} 2023, Tianjin, China, April 17-20, 2023},
  publisher = {Springer},
  year      = {2023}
}
```
## quick start
```latex
python main.py --dataset ICEWS14-few --max_neighbor 50 --process_step 4 --num_attention_heads 2
python main.py --dataset ICEWS05-15-few --max_neighbor 50 --process_step 4 --num_attention_heads 2
python main.py --dataset ICEWS18-few --max_neighbor 100 --process_step 5 --num_attention_heads 4
```
