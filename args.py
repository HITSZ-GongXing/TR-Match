import argparse
import os


def read_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ICEWS14-few", type=str) # 数据集路径
    parser.add_argument("--num_days", default=365, type=int)  #
    parser.add_argument("--embed_dim", default=100, type=int) # 实体与关系的维度
    # parser.add_argument("--train_few", default=3, type=int) #
    parser.add_argument("--few", default=3, type=int) # few-shot
    parser.add_argument("--process_step", default=5, type=int)  # few-shot
    parser.add_argument("--hidden_size", default=200, type=int)  # 注意力模型hidden_size
    parser.add_argument("--batch_size", default=128, type=int) # query-set个数
    parser.add_argument("--neg_num", default=1, type=int)
    parser.add_argument("--random_embed", action='store_true') # 选择随机嵌入还是Pre-train的Embedding
    parser.add_argument("--lr", default=0.001, type=float) # lr
    parser.add_argument("--margin", default=10.0, type=float) # lambda
    parser.add_argument("--gamma", default=0.5, type=float)  # gamma
    parser.add_argument("--dropout", default=0.2, type=float) #
    parser.add_argument("--dropout_layers", default=0.2, type=float)
    parser.add_argument("--dropout_neighbors", default=0.0, type=float)

    parser.add_argument("--process_steps", default=2, type=int)
    parser.add_argument("--log_every", default=100, type=int) # log的周期
    parser.add_argument("--eval_every", default=1000, type=int) # eval的周期
    parser.add_argument("--fine_tune", action='store_true') # 使用Pre-train的Embedding是否微调
    parser.add_argument("--max_neighbor", default=100, type=int) # 最大的邻居数
    parser.add_argument("--no_meta", action='store_true')
    parser.add_argument("--test", action='store_true') # 测试程序
    parser.add_argument("--dev", action='store_true')
    parser.add_argument("--embed_model", default='TransE', type=str) # 选择Pre-train的模型
    parser.add_argument("--prefix", default='intial', type=str) # 写入的目标文件夹

    parser.add_argument("--seed", default='19950922', type=int) # 随机种子
    parser.add_argument("--loss", default='origin', type=str)

    parser.add_argument("--num_attention_heads", default=2, type=int) # 多头注意力模型
    parser.add_argument("--num_transformer_layers", default=3, type=int)
    parser.add_argument("--num_transformer_heads", default=4, type=int)
    parser.add_argument("--warm_up_step", default=10000, type=int)
    parser.add_argument("--max_batches", default=500000, type=int) # 最大限制的训练任务数
    parser.add_argument("--weight_decay", default=0.0001, type=float)
    parser.add_argument("--grad_clip", default=5.0, type=float) # max norm of the gradients

    args = parser.parse_args()
    if not os.path.exists('models'):
        os.mkdir('models')
    args.save_path = 'models/' + args.prefix

    return args


if __name__ == '__main__':
    args = read_options()