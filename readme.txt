  推荐运行顺序

  source .venv/bin/activate

  # 1. 先评估基线（下载模型约 3GB，首次较慢）
  python eval_baseline.py

  # 2. 微调训练
  python train.py

  # 3. 评估微调后模型，与基线对比
  python inference.py --eval