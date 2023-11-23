import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go


# Đọc dữ liệu từ file CSV
@st.cache
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Tính giá cả trung bình của từng quận
def calculate_district_prices(data):
    district_prices = data.groupby('Quận')['Giá (triệu đồng/m2)'].mean().reset_index()
    district_prices = district_prices.rename(columns={'Giá (triệu đồng/m2)': 'Giá Quận'})
    return district_prices

# Tạo và huấn luyện mô hình dự đoán
def train_model(data):
    X = data[['Số tầng', 'Số phòng ngủ', 'Diện tích', 'Giá Quận']]
    y = data['Giá (triệu đồng/m2)']  # Chọn biến mục tiêu
    model = LinearRegression()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Đánh giá độ chính xác
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    
    st.write("Đánh giá độ chính xác của mô hình:")
    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    st.write(f"R-squared (R²): {r2:.2f}")

    return model

def plot_price_and_features(input_data, prediction, similar_houses, chart_type):
    if chart_type == "Scatter Plot":
        fig = px.scatter(similar_houses, x='Diện tích', y='Giá (triệu đồng/m2)', 
                         title='Biểu đồ Giá Cả', labels={'Diện tích': 'Diện tích (m2)', 'Giá (triệu đồng/m2)': 'Giá (triệu đồng/m2)'})

        fig.add_trace(go.Scatter(x=[input_data[0][2]], y=[prediction], mode='markers', marker=dict(color='red', size=12),
                                 name='Dự Đoán', text=f'Giá Dự Đoán: {prediction:.2f} triệu đồng/m2',
                                 hovertemplate='Diện tích (m2): %{x} m2<br>Giá: %{y:.2f} triệu đồng/m2'))
        

        # Hiển thị tên của từng điểm khi di chuột qua
        fig.update_traces(textposition='top center')

        st.plotly_chart(fig)
    elif chart_type == "Bar Chart":
        fig = px.bar(similar_houses, x='Địa chỉ', y='Giá (triệu đồng/m2)', 
                     title='Biểu đồ Giá Cả', labels={'Địa chỉ': 'Địa chỉ', 'Giá (triệu đồng/m2)': 'Giá (triệu đồng/m2)'})
        
        st.plotly_chart(fig)

# Load dữ liệu
data = load_data("file_new/du_lieu_lam_sach.csv")

# Tính giá cả trung bình của từng quận
district_prices = calculate_district_prices(data)

# Kết hợp thông tin về giá cả của quận vào dữ liệu chính
data = data.merge(district_prices, on='Quận')

# Huấn luyện mô hình và đánh giá độ chính xác
model = train_model(data)

# Thêm các trường nhập liệu
st.sidebar.header("Nhập thông tin để dự đoán giá nhà")
num_floors = st.sidebar.number_input("Số tầng", min_value=1, max_value=11)
num_bedrooms = st.sidebar.number_input("Số phòng ngủ", min_value=1, max_value=11)
area = st.sidebar.number_input("Diện tích (m2)", min_value=20, max_value=200)

# Thêm trường nhập liệu chọn quận
selected_district = st.sidebar.selectbox("Chọn Quận", data['Quận'].unique())

chart_type = st.sidebar.radio("Chọn loại biểu đồ", ["Scatter Plot", "Bar Chart"])

if st.sidebar.button("Dự Đoán"):
    district_price = data[data['Quận'] == selected_district]['Giá Quận'].values[0]

    input_data = [[num_floors, num_bedrooms, area, district_price]]
    prediction = model.predict(input_data)
    st.sidebar.write(f"Giá Nhà Dự Đoán: {prediction[0]:.2f} triệu đồng/m2")

    filtered_data = data[(data['Quận'] == selected_district) & 
                         (data['Diện tích'] >= area - 2) & (data['Diện tích'] <= area + 2) & 
                         (data['Số tầng'] >= num_floors - 1) & (data['Số tầng'] <= num_floors + 1) & 
                         (data['Số phòng ngủ'] >= num_bedrooms - 1) & (data['Số phòng ngủ'] <= num_bedrooms + 1)]

    price_tolerance = 5.0

    similar_houses = filtered_data[
        (filtered_data['Giá (triệu đồng/m2)'] >= prediction[0] - price_tolerance) &
        (filtered_data['Giá (triệu đồng/m2)'] <= prediction[0] + price_tolerance)
    ]

    st.sidebar.write(f"Các ngôi nhà phù hợp trong quận {selected_district} với diện tích gần {area} m2:")
    st.sidebar.write(similar_houses[['Địa chỉ', 'Giá (triệu đồng/m2)', 'Diện tích', 'Số phòng ngủ', 'Số tầng']])
    
    # Gọi hàm để vẽ biểu đồ
    plot_price_and_features(input_data, prediction[0], similar_houses, chart_type)

    if similar_houses.empty:
        st.sidebar.write("Lưu ý: Không có ngôi nhà phù hợp với dự đoán trong khu vực đã lựa chọn.")
    elif 'Giá (triệu đồng/m2)' in data.columns:  
        if prediction[0] >= data['Giá (triệu đồng/m2)'].mean():
            st.sidebar.write("Lưu ý: Giá dự đoán ở mức cao hơn giá trung bình khu vực đã lựa chọn.")
        else:
            st.sidebar.write("Lưu ý: Giá dự đoán ở mức thấp hơn giá trung bình khu vực đã lựa chọn.")
    else:
        st.sidebar.write("Lưu ý: Không có dữ liệu giá trung bình khu vực đã lựa chọn.")
