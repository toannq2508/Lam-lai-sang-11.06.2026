# THÀNH PHẦN KHỞI TẠO ĐẦU TIÊN
import streamlit as st
st.set_page_config(layout="wide", page_title="Credit Risk & Fraud Detection", page_icon="🕵️‍♂️")

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# ==========================================
# CÁC HÀM DÙNG CHUNG (CACHE)
# ==========================================
@st.cache_data
def load_data(file_bytes, file_name):
    """Đọc dữ liệu từ file upload (bytes) để cache hợp lệ."""
    if file_name.endswith('.csv'):
        df = pd.read_csv(file_bytes)
    elif file_name.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(file_bytes)
    else:
        return None
    return df

# Cấu hình các biến đầu vào theo tập dữ liệu của anh
FEATURES = [f'X_{i}' for i in range(1, 15)]  # Sinh tự động X_1 đến X_14
TARGET = 'default'

# 🎨 BẢNG MÀU TRENDY (Xu hướng Modern Tech)
COLOR_SAFE = '#00CECB'  # Xanh Cyan (An toàn)
COLOR_RISK = '#FF5E5B'  # Đỏ Coral (Cảnh báo rủi ro)
COLOR_PALETTE = [COLOR_SAFE, COLOR_RISK, '#FFED66', '#5C80BC', '#8C52FF']

# ==========================================
# THÀNH PHẦN 1: SIDEBAR - CẤU HÌNH BỀN VỮNG
# ==========================================
with st.sidebar:
    st.header("⚙️ Cấu hình & Tải dữ liệu")
    
    uploaded_file = st.file_uploader("Tải lên file dữ liệu (.csv, .xlsx)", type=['csv', 'xlsx'])
    
    st.divider()
    
    model_choice = st.selectbox(
        "Lựa chọn Mô hình",
        options=["Random Forest", "Logistic Regression"],
        help="Thuật toán học máy dùng để phân loại."
    )
    
    st.subheader("Tham số mô hình AI")
    if model_choice == "Random Forest":
        n_estimators = st.slider("Số lượng cây (n_estimators)", min_value=10, max_value=200, value=50, step=10)
        max_depth = st.slider("Độ sâu tối đa (max_depth)", min_value=2, max_value=20, value=10, step=1)
        random_state = st.number_input("Random State", value=42, step=1)
    else:
        max_iter = st.number_input("Số vòng lặp tối đa (max_iter)", min_value=100, max_value=1000, value=200, step=50)
        C_param = st.slider("Tham số điều chuẩn (C)", min_value=0.01, max_value=10.0, value=1.0, step=0.1)
        random_state = st.number_input("Random State", value=42, step=1)
    
    test_size = st.slider("Tỷ lệ tập kiểm định (test_size)", min_value=0.1, max_value=0.5, value=0.2, step=0.05)

    st.divider()
    train_button = st.button("🚀 Huấn luyện Mô hình", type="primary", use_container_width=True)

# ==========================================
# THÀNH PHẦN 2: HEADER - VÙNG ĐỊNH HƯỚNG
# ==========================================
st.title("🕵️‍♂️ Ứng dụng Chấm điểm Rủi ro (Risk Scoring)")
st.caption("Ứng dụng AI phân tích và phát hiện rủi ro hồ sơ/giao dịch dựa trên dữ liệu đầu vào. Giao diện được thiết kế theo xu hướng công nghệ hiện đại.")

if uploaded_file is None:
    st.info("👈 Vui lòng tải lên file dữ liệu ở thanh bên trái để bắt đầu.")
    st.stop()

# Đọc dữ liệu
try:
    df_raw = load_data(uploaded_file, uploaded_file.name)
    if df_raw is None:
        st.error("Định dạng file không được hỗ trợ!")
        st.stop()
except Exception as e:
    st.error(f"Lỗi đọc file: {e}")
    st.stop()

# Kiểm tra schema dữ liệu
missing_cols = [col for col in FEATURES + [TARGET] if col not in df_raw.columns]
if missing_cols:
    st.error(f"File tải lên thiếu các cột bắt buộc: {', '.join(missing_cols)}")
    st.stop()

st.caption(f"📁 Đang dùng tệp: **{uploaded_file.name}** | 📊 Kích thước: {df_raw.shape[0]:,} dòng, {df_raw.shape[1]} cột")
st.divider()

# ==========================================
# KHỐI XỬ LÝ HUẤN LUYỆN (Chạy khi bấm nút)
# ==========================================
if train_button:
    with st.spinner("Đang tiền xử lý và huấn luyện mô hình..."):
        # 1. Tiền xử lý
        df = df_raw.copy()
        df = df.dropna(subset=FEATURES + [TARGET])
        
        # Xử lý tự động biến phân loại
        label_encoders = {}
        for col in FEATURES:
            if df[col].dtype == 'object':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                label_encoders[col] = le
        
        X = df[FEATURES]
        y = df[TARGET]
        
        # 2. Chia tập
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        
        # 3. Chuẩn hóa dữ liệu
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 4. Huấn luyện
        if model_choice == "Random Forest":
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=-1)
        else:
            model = LogisticRegression(max_iter=max_iter, C=C_param, random_state=random_state)
            
        model.fit(X_train_scaled, y_train)
        
        # 5. Đánh giá nhanh
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None
        
        # 6. Lưu vào session_state
        st.session_state['model'] = model
        st.session_state['scaler'] = scaler
        st.session_state['label_encoders'] = label_encoders
        st.session_state['y_test'] = y_test
        st.session_state['y_pred'] = y_pred
        st.session_state['y_proba'] = y_proba
        st.session_state['model_name'] = model_choice
        
    st.success("✅ Huấn luyện thành công! Xem chi tiết ở các Tab bên dưới.")

# ==========================================
# CÁC TAB HIỂN THỊ NỘI DUNG CHÍNH
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Tổng quan Dữ liệu", 
    "✨ Trực quan hóa Dữ liệu", 
    "🎯 Kết quả Kiểm định", 
    "🚀 Sử dụng Mô hình"
])

# ------------------------------------------
# THÀNH PHẦN 3: TỔNG QUAN DỮ LIỆU
# ------------------------------------------
with tab1:
    col1, col2, col3 = st.columns(3)
    col1.metric("Số lượng bản ghi", f"{df_raw.shape[0]:,}")
    col2.metric("Số lượng đặc trưng (X)", f"{len(FEATURES)}")
    col3.metric(f"Tỷ lệ {TARGET} (Rủi ro/Gian lận)", f"{(df_raw[TARGET].mean() * 100):.2f}%")
    
    st.subheader("Xem trước Dữ liệu")
    with st.container(height=300):
        st.dataframe(df_raw.head(100), use_container_width=True)
        
    st.subheader("Thống kê Mô tả (Biến đầu vào)")
    st.dataframe(df_raw[FEATURES].describe(), use_container_width=True)

# ------------------------------------------
# THÀNH PHẦN 4: TRỰC QUAN HÓA DỮ LIỆU (CẬP NHẬT MÀU SẮC ĐẸP MẮT)
# ------------------------------------------
with tab2:
    st.subheader("Biểu đồ phân phối theo phong cách Modern Tech")
    
    # Đảm bảo target được coi là string/category để lên màu rời rạc đẹp nhất
    df_plot = df_raw.copy()
    df_plot[TARGET] = df_plot[TARGET].astype(str)
    
    col_v1, col_v2 = st.columns(2)
    
    with col_v1:
        # Biểu đồ Donut chart sành điệu
        fig1 = px.pie(df_plot, names=TARGET, title=f"Tỷ lệ phân phối biến mục tiêu ({TARGET})", 
                      hole=0.5, color_discrete_sequence=[COLOR_SAFE, COLOR_RISK])
        fig1.update_traces(textposition='inside', textinfo='percent+label', hoverinfo='label+percent',
                           marker=dict(line=dict(color='#000000', width=1)))
        fig1.update_layout(showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)
        
    with col_v2:
        if df_plot['X_1'].nunique() < 20:
            fig2 = px.histogram(df_plot, x='X_1', title="Phân phối của biến X_1 (Phân loại)", 
                                color=TARGET, barmode='group', color_discrete_sequence=[COLOR_SAFE, COLOR_RISK])
        else:
            fig2 = px.histogram(df_plot, x='X_1', title="Phân phối của biến X_1 (Liên tục)", 
                                color=TARGET, nbins=40, opacity=0.75, barmode='overlay', 
                                color_discrete_sequence=[COLOR_SAFE, COLOR_RISK])
        fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)", yaxis_title="Số lượng", xaxis_title="Giá trị X_1")
        st.plotly_chart(fig2, use_container_width=True)

    col_v3, col_v4 = st.columns(2)
    
    with col_v3:
        if df_plot['X_2'].nunique() < 20:
            fig3 = px.histogram(df_plot, x='X_2', title="Phân phối của biến X_2 (Phân loại)", 
                                color=TARGET, barmode='group', color_discrete_sequence=[COLOR_SAFE, COLOR_RISK])
        else:
            fig3 = px.histogram(df_plot, x='X_2', title="Phân phối của biến X_2 (Liên tục)", 
                                color=TARGET, nbins=40, opacity=0.75, barmode='overlay',
                                color_discrete_sequence=[COLOR_SAFE, COLOR_RISK])
        fig3.update_layout(plot_bgcolor="rgba(0,0,0,0)", yaxis_title="Số lượng", xaxis_title="Giá trị X_2")
        st.plotly_chart(fig3, use_container_width=True)
        
    with col_v4:
        if pd.api.types.is_numeric_dtype(df_raw['X_1']):
            fig4 = px.box(df_plot, x=TARGET, y='X_1', title=f"Sự phân hóa của X_1 theo {TARGET}", 
                          color=TARGET, color_discrete_sequence=[COLOR_SAFE, COLOR_RISK])
            fig4.update_layout(plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("Biểu đồ Boxplot ưu tiên biến liên tục, X_1 đang là phân loại.")

# ------------------------------------------
# THÀNH PHẦN 5: KẾT QUẢ KIỂM ĐỊNH MÔ HÌNH (CẬP NHẬT MÀU SẮC ĐẸP MẮT)
# ------------------------------------------
with tab3:
    if 'model' not in st.session_state or 'y_test' not in st.session_state:
        st.info("💡 Mô hình chưa được huấn luyện. Vui lòng thiết lập tham số ở thanh bên trái và bấm 'Huấn luyện Mô hình'.")
    else:
        st.subheader(f"Chỉ tiêu kiểm định: {st.session_state['model_name']}")
        
        y_test = st.session_state['y_test']
        y_pred = st.session_state['y_pred']
        y_proba = st.session_state['y_proba']
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
        c2.metric("Precision", f"{precision_score(y_test, y_pred, zero_division=0):.4f}")
        c3.metric("Recall", f"{recall_score(y_test, y_pred, zero_division=0):.4f}")
        c4.metric("F1 Score", f"{f1_score(y_test, y_pred, zero_division=0):.4f}")
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("**Ma trận nhầm lẫn (Confusion Matrix)**")
            cm = confusion_matrix(y_test, y_pred)
            # Dùng dải màu Magma rực rỡ, thịnh hành
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Magma', aspect="auto",
                               labels=dict(x="Dự báo (Predicted)", y="Thực tế (Actual)", color="Số lượng"),
                               x=['Bình thường (0)', 'Rủi ro (1)'], y=['Bình thường (0)', 'Rủi ro (1)'])
            fig_cm.update_layout(plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_cm, use_container_width=True)
            
        with col_m2:
            st.markdown("**Đường cong ROC (ROC Curve)**")
            if y_proba is not None:
                from sklearn.metrics import roc_curve
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                auc_score = roc_auc_score(y_test, y_proba)
                
                # Tạo ROC curve với hiệu ứng vùng fill bắt mắt
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', line=dict(color=COLOR_RISK, width=3),
                                             fill='tozeroy', fillcolor='rgba(255, 94, 91, 0.2)', name=f'ROC (AUC = {auc_score:.4f})'))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(color='gray', width=2, dash='dash'), showlegend=False))
                
                fig_roc.update_layout(title=f"Đường cong ROC (AUC = {auc_score:.4f})",
                                      xaxis_title='False Positive Rate', yaxis_title='True Positive Rate',
                                      plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
                st.plotly_chart(fig_roc, use_container_width=True)
            else:
                st.warning("Mô hình không hỗ trợ xuất xác suất (predict_proba) để vẽ ROC.")

# ------------------------------------------
# THÀNH PHẦN 6: SỬ DỤNG MÔ HÌNH
# ------------------------------------------
with tab4:
    if 'model' not in st.session_state or 'scaler' not in st.session_state:
        st.info("💡 Vui lòng huấn luyện mô hình trước khi sử dụng để dự báo.")
    else:
        st.subheader("Dự báo dữ liệu mới")
        mode = st.radio("Chọn phương thức nhập dữ liệu:", ["✏️ Nhập thủ công từng hồ sơ", "📁 Tải file dự báo hàng loạt"])
        
        model = st.session_state['model']
        scaler = st.session_state['scaler']
        label_encoders = st.session_state['label_encoders']
        
        if mode == "✏️ Nhập thủ công từng hồ sơ":
            with st.form("predict_form"):
                st.markdown("**Nhập thông tin hồ sơ (X_1 đến X_14):**")
                
                cols = st.columns(3)
                input_data = {}
                
                for i, col_name in enumerate(FEATURES):
                    target_col = cols[i % 3]
                    with target_col:
                        if col_name in label_encoders:
                            options = list(label_encoders[col_name].classes_)
                            input_data[col_name] = st.selectbox(f"{col_name}", options=options)
                        else:
                            default_val = float(df_raw[col_name].median()) if col_name in df_raw else 0.0
                            input_data[col_name] = st.number_input(f"{col_name}", value=default_val)
                
                submit_pred = st.form_submit_button("🔍 Dự báo Rủi ro")
                
                if submit_pred:
                    input_df = pd.DataFrame([input_data])
                    
                    for c in FEATURES:
                        if c in label_encoders:
                            input_df[c] = label_encoders[c].transform(input_df[c].astype(str))
                    
                    input_scaled = scaler.transform(input_df)
                    pred = model.predict(input_scaled)[0]
                    prob = model.predict_proba(input_scaled)[0][1] if hasattr(model, "predict_proba") else None
                    
                    st.divider()
                    if pred == 1:
                        st.error(f"🚨 **CẢNH BÁO RỦI RO CAO:** Hồ sơ bị xếp vào nhóm Default=1! (Xác suất rủi ro: {prob*100:.2f}%)")
                    else:
                        st.success(f"✅ HỒ SƠ AN TOÀN: Xếp vào nhóm Default=0. (Xác suất rủi ro: {prob*100:.2f}%)")
                        
        else:
            upload_test = st.file_uploader("Tải file CSV/Excel cần dự báo", type=['csv', 'xlsx'], key="upload_test")
            if upload_test is not None:
                df_test_raw = load_data(upload_test, upload_test.name)
                
                missing = [c for c in FEATURES if c not in df_test_raw.columns]
                if missing:
                    st.error(f"File thiếu các cột: {', '.join(missing)}")
                else:
                    st.success("File hợp lệ! Bấm nút dưới đây để dự báo.")
                    if st.button("Dự báo Hàng loạt"):
                        df_pred = df_test_raw.copy()
                        
                        for c in FEATURES:
                            if c in label_encoders:
                                known_classes = list(label_encoders[c].classes_)
                                df_pred[c] = df_pred[c].apply(lambda x: x if x in known_classes else known_classes[0])
                                df_pred[c] = label_encoders[c].transform(df_pred[c].astype(str))
                        
                        X_test_batch = df_pred[FEATURES]
                        X_test_batch_scaled = scaler.transform(X_test_batch)
                        preds = model.predict(X_test_batch_scaled)
                        
                        df_test_raw[f'Prediction_{TARGET}'] = preds
                        
                        if hasattr(model, "predict_proba"):
                            probs = model.predict_proba(X_test_batch_scaled)[:, 1]
                            df_test_raw[f'Probability_{TARGET}'] = np.round(probs, 4)
                            
                        with st.container(height=300):
                            st.dataframe(df_test_raw)
                            
                        csv = df_test_raw.to_csv(index=False).encode('utf-8-sig')
                        st.download_button(
                            label="📥 Tải Bảng Kết quả (CSV)",
                            data=csv,
                            file_name="predicted_results.csv",
                            mime="text/csv"
                        )
