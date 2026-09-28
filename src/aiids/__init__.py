"""AI-IDS —— 基于机器学习的网络入侵检测与实时告警系统。

模块划分：
    aiids.config      配置加载
    aiids.data        数据加载、清洗、划分
    aiids.features    特征工程（编码、标准化）
    aiids.models      模型训练与评估
    aiids.imbalance   类别不平衡处理
    aiids.explain     特征重要性与 SHAP
    aiids.generalize  未知攻击泛化实验
    aiids.flow        tshark/pcap 流量提取
    aiids.alert       告警引擎
    aiids.api         FastAPI 推理服务
    aiids.db          SQLite 存储
"""

__version__ = "1.0.0"
