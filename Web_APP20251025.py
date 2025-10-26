import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib

# ---------------------- 1. 基础配置 ----------------------
plt.rcParams["font.family"] = ["Times New Roman", "SimHei"]
plt.rcParams['axes.unicode_minus'] = False

# ---------------------- 2. 自定义CSS：统一样式 & 蓝色按钮 ----------------------
st.markdown("""
<style>
body {
    background-color: #f5f7fa;
    font-family: "Helvetica Neue", Arial, sans-serif;
}

.card {
    background-color: white;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    padding: 20px;
    margin-bottom: 20px;
}

.section-title {
    font-size: 18px;
    font-weight: bold;
    color: #2c3e50;
    border-bottom: 2px solid #3498db;
    padding-bottom: 10px;
    margin-bottom: 15px;
}

.label-col {
    text-align: left !important;
    width: 180px;
    padding-right: 10px;
    font-size: 13px;
}

.input-col {
    flex: 1;
}

/* 全局文本左对齐 */
div[class*="stText"], div[class*="stNumberInput"], div[class*="stSelectbox"] {
    text-align: left !important;
}
/* 蓝色预测按钮 */
.stButton>button {
    background-color: #3498db !important;
    color: white !important;
    text-align: center !important;
    border-radius: 6px !important;
    padding: 10px 20px !important;
    font-size: 16px !important;
    border: 2px solid white !important;   /* 添加白色边框 */
}
.stButton>button:hover {
    background-color: #2980b9 !important;
            
}
</style>
""", unsafe_allow_html=True)


# ---------------------- 3. 加载模型 & 定义特征范围 ----------------------
# 加载XGBoost模型
try:
    model = joblib.load('xgboost_model.pkl')
    st.success("XGBoost model loaded successfully!")
except FileNotFoundError:
    st.error("Model file not found! Ensure 'xgboost_model.pkl' is in the current directory.")
    st.stop()

# 定义特征范围
feature_ranges = {
    '年龄（岁）': {"type": "numerical", "min": 18.0, "max": 90.0, "default": 55.0},
    '体重指数（BMI）': {"type": "numerical", "min": 18.0, "max": 35.0, "default": 24.0},
    '收缩压（mmHg）': {"type": "numerical", "min": 90.0, "max": 160.0, "default": 120.0},
    '舒张压（mmHg）': {"type": "numerical", "min": 60.0, "max": 100.0, "default": 80.0},
    '空腹血糖（mmol/L）': {"type": "numerical", "min": 3.9, "max": 11.1, "default": 5.2},
    '肌酐（μmol/L）': {"type": "numerical", "min": 40.0, "max": 133.0, "default": 70.0},
    '尿素氮（mmol/L）': {"type": "numerical", "min": 2.5, "max": 8.2, "default": 5.0},
    '血红蛋白（g/L）': {"type": "numerical", "min": 110.0, "max": 160.0, "default": 130.0},
    '白细胞计数（×10^9/L）': {"type": "numerical", "min": 3.5, "max": 9.5, "default": 6.0},
    '血小板计数（×10^9/L）': {"type": "numerical", "min": 125.0, "max": 350.0, "default": 200.0},
    '总胆固醇（mmol/L）': {"type": "numerical", "min": 3.1, "max": 6.2, "default": 4.5},
    '甘油三酯（mmol/L）': {"type": "numerical", "min": 0.5, "max": 3.0, "default": 1.5},
    '高密度脂蛋白（mmol/L）': {"type": "numerical", "min": 0.9, "max": 1.5, "default": 1.2},
    '低密度脂蛋白（mmol/L）': {"type": "numerical", "min": 2.0, "max": 4.1, "default": 3.0},
    '钠（mmol/L）': {"type": "numerical", "min": 135.0, "max": 145.0, "default": 140.0},
    '钾（mmol/L）': {"type": "numerical", "min": 3.5, "max": 5.5, "default": 4.2},
    '氯（mmol/L）': {"type": "numerical", "min": 95.0, "max": 105.0, "default": 100.0},
    '白蛋白（g/L）': {"type": "numerical", "min": 35.0, "max": 50.0, "default": 40.0},
    '性别': {"type": "categorical", "options": [0, 1], "label": ["Male (0)", "Female (1)"]},
    '糖尿病分级': {"type": "categorical", "options": [0, 1, 2], "label": ["None (0)", "Mild (1)", "Severe (2)"]},
    '高血压分级': {"type": "categorical", "options": [0, 1, 2], "label": ["None (0)", "Grade 1 (1)", "Grade 2 (2)"]}
}


# ---------------------- 4. 页面结构 ----------------------
st.title("Clinical Disease Risk Prediction", anchor=False)
st.markdown("Enter clinical indicators to predict disease risk and view feature contributions.")

# 临床指标模块（3列布局）
with st.container():
    st.markdown('<div class="card"><h3 class="section-title">Clinical Indicators</h3>', unsafe_allow_html=True)
    cols = st.columns(3)
    feature_values = []
    feature_names = list(feature_ranges.keys())

    for idx, (feature, props) in enumerate(feature_ranges.items()):
        with cols[idx % 3]:
            st.markdown(f'<div style="display: flex; align-items: center; margin-bottom: 15px;"><div class="label-col">{feature}</div><div class="input-col">', unsafe_allow_html=True)
            if props["type"] == "numerical":
                step = 0.1 if feature in ["体重指数（BMI）", "空腹血糖（mmol/L）"] else 1.0
                fmt = "%.1f" if feature in ["体重指数（BMI）", "空腹血糖（mmol/L）"] else "%.0f"
                value = st.number_input(
                    feature,
                    min_value=float(props["min"]),
                    max_value=float(props["max"]),
                    value=float(props["default"]),
                    step=step,
                    format=fmt,
                    label_visibility="collapsed"
                )
            else:
                value = st.selectbox(
                    feature,
                    options=props["options"],
                    format_func=lambda x: props["label"][props["options"].index(x)],
                    label_visibility="collapsed"
                )
            feature_values.append(value)
            st.markdown('</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------- 5. 预测与SHAP可视化 ----------------------
if st.button("Predict Risk", type="primary", use_container_width=True, key="predict_btn"):
    input_data = pd.DataFrame([feature_values], columns=feature_names)
    
    # 模型预测
    pred_class = model.predict(input_data)[0]
    pred_proba = model.predict_proba(input_data)[0]
    risk_prob = pred_proba[1] * 100

    # 计算风险等级（关键：先定义status和color）
    if risk_prob < 30:
        status, color = "Low Risk", "green"
    elif 30 <= risk_prob < 70:
        status, color = "Medium Risk", "orange"
    else:
        status, color = "High Risk", "red"

    # 计算SHAP值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)
    base_value = explainer.expected_value
    single_shap = shap_values[0]

    # 将结果存入session_state
    st.session_state.pred_results = {
        "status": status,
        "color": color,
        "risk_prob": risk_prob,
        "no_risk_prob": pred_proba[0] * 100,
        "single_shap": single_shap,
        "feature_names": feature_names,
        "feature_values": feature_values,
        "base_value": base_value  # 保存base_value用于后续绘图
    }

# 显示预测结果
if "pred_results" in st.session_state:
    res = st.session_state.pred_results
    st.markdown("### Prediction Result")
    st.markdown(
        f"<div style='font-family: Times New Roman; font-size:14px; color:{res['color']}; font-weight:bold; margin:12px 0'>"
        f"Disease Risk Level: {res['status']}<br>"
        f"Risk Probability: {res['risk_prob']:.2f}%<br>"
        f"No-Risk Probability: {res['no_risk_prob']:.2f}%"
        "</div>",
        unsafe_allow_html=True
    )

    # 显示SHAP瀑布图
    st.markdown("### SHAP Waterfall Plot (Feature Contribution)")
    st.markdown("Blue = Reduce risk, Red = Increase risk, Length = Contribution degree (Top 10 features)")
    
    single_data = pd.DataFrame([res['feature_values']], columns=res['feature_names']).iloc[0].values
    shap_exp = shap.Explanation(
        values=res['single_shap'],
        base_values=res['base_value'],  # 使用保存的base_value
        data=single_data,
        feature_names=res['feature_names']
    )

    plt.figure(figsize=(12, 8))
    shap.plots.waterfall(shap_exp, max_display=10, show=False)
    plt.tight_layout()
    plt.savefig("shap_waterfall.png", dpi=300, bbox_inches='tight')
    st.image("shap_waterfall.png", use_column_width=True)

    # 显示所有特征的SHAP贡献值
    if st.checkbox("Show all features' SHAP values", key="show_shap"):
        shap_df = pd.DataFrame({
            "Feature (Chinese)": res['feature_names'],
            "Input Value": res['feature_values'],
            "SHAP Value (Contribution)": res['single_shap'].round(4)
        })
        shap_df["Absolute Contribution"] = shap_df["SHAP Value (Contribution)"].abs()
        shap_df_sorted = shap_df.sort_values("Absolute Contribution", ascending=False).drop("Absolute Contribution", axis=1)
        st.dataframe(shap_df_sorted, use_container_width=True)  