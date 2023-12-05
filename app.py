import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go   #đồ thị tương tác, xử lí các sự kiện


# Đọc dữ liệu từ file CSV
@st.cache
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Tính giá cả trung bình của từng quận
def calculate_district_prices(data):
    district_prices = data.groupby('Quận')['Giá (triệu đồng/m2)'].mean().reset_index()
    district_prices = district_prices.rename(columns={'Giá (triệu đồng/m2)': 'Giá Quận'})  # đổi cột thành dataframe
    return district_prices
#reset_index():Chuyển đổi kết quả từ Series sang DataFrame và đặt lại chỉ số (index).Kết quả là một DataFrame có cột 'Quận' và 'Giá (triệu đồng/m2)'.

# Tạo và huấn luyện mô hình dự đoán
def train_model(data):
    X = data[['Số tầng', 'Số phòng ngủ', 'Diện tích', 'Giá Quận']]
    y = data['Giá (triệu đồng/m2)']  # Chọn biến mục tiêu
    model = LinearRegression()  #hồi quy tuyến tính
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    #đánh giá mô hình
    r2 = r2_score(y_test, y_pred)
    
    st.write("Đánh giá độ chính xác của mô hình:")
    st.write(f"R-squared (R²): {r2:.2f}")

    return model

def plot_price_and_features(input_data, prediction, similar_houses, chart_type):
    if chart_type == "Scatter Plot":
        fig = px.scatter(similar_houses, x='Diện tích', y='Giá (triệu đồng/m2)', 
                         title='Biểu đồ Giá Cả', labels={'Diện tích': 'Diện tích (m2)', 'Giá (triệu đồng/m2)': 'Giá (triệu đồng/m2)'})

        fig.add_trace(go.Scatter(x=[input_data[0][2]], y=[prediction[0]], mode='markers', marker=dict(color='red', size=12),
                                 name='Dự Đoán', text=f'Giá Dự Đoán: {prediction[0]:.2f} triệu đồng/m2',
                                 hovertemplate='Diện tích (m2): %{x} m2<br>Giá: %{y:.2f} triệu đồng/m2'))

        # Hiển thị tên của từng điểm khi di chuột qua
        fig.update_traces(textposition='top center')
        st.plotly_chart(fig)

        #prediction[0] là giá dự đoán của mô hình
    elif chart_type == "Pie Chart":
        labels = ['Giá Dự Đoán', 'Giá Trung Bình']
        values = [prediction[0], data['Giá (triệu đồng/m2)'].mean()]
        fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
        st.plotly_chart(fig)


data = load_data("file_new/du_lieu_lam_sach.csv")

# Tính giá cả trung bình của từng quận
district_prices = calculate_district_prices(data)

# Kết hợp thông tin về giá cả của quận vào dữ liệu chính
data = data.merge(district_prices, on='Quận')

# Huấn luyện mô hình và đánh giá độ chính xác
model = train_model(data)

# Thêm các trường nhập liệu
st.sidebar.header("Nhập thông tin để dự đoán giá nhà")
num_floors = st.sidebar.number_input("Số tầng", min_value=1, max_value=15)
num_bedrooms = st.sidebar.number_input("Số phòng ngủ", min_value=1, max_value=15)
area = st.sidebar.number_input("Diện tích (m2)", min_value=20, max_value=500)


selected_district = st.sidebar.selectbox("Chọn Quận", data['Quận'].unique())
selected_housing_type = st.sidebar.selectbox("Chọn Loại hình nhà ở", data['Loại hình nhà ở'].unique())


chart_type = st.sidebar.radio("Chọn loại biểu đồ", ["Scatter Plot", "Pie Chart"])


if st.sidebar.button("Dự Đoán"):
    district_price = data[data['Quận'] == selected_district]['Giá Quận'].values[0]

    input_data = [[num_floors, num_bedrooms, area, district_price]]
    prediction = model.predict(input_data)  
    st.markdown(f"<p style='font-size:24px; font-weight:bold; color:green;'>Giá Nhà Dự Đoán: {prediction[0]:.2f} triệu đồng/m2</p>", unsafe_allow_html=True)


    filtered_data = data[(data['Quận'] == selected_district) & 
                         (data['Loại hình nhà ở'] == selected_housing_type) &
                         (data['Diện tích'] >= area - 2) & (data['Diện tích'] <= area + 2) & 
                         (data['Số tầng'] >= num_floors - 1) & (data['Số tầng'] <= num_floors + 1) & 
                         (data['Số phòng ngủ'] >= num_bedrooms - 1) & (data['Số phòng ngủ'] <= num_bedrooms + 1)]

    price_tolerance = 5.0 #chênh lệch giá

    similar_houses = filtered_data[
        (filtered_data['Giá (triệu đồng/m2)'] >= prediction[0] - price_tolerance) &
        (filtered_data['Giá (triệu đồng/m2)'] <= prediction[0] + price_tolerance)
    ]

    st.sidebar.write(f"Các ngôi nhà phù hợp trong quận {selected_district} với diện tích gần {area} m2 và loại hình nhà ở {selected_housing_type}:")
    st.sidebar.write(similar_houses[['Địa chỉ', 'Giá (triệu đồng/m2)', 'Diện tích', 'Số phòng ngủ', 'Số tầng', 'Loại hình nhà ở']])
    
    # Gọi hàm để vẽ biểu đồ
    plot_price_and_features(input_data, prediction, similar_houses, chart_type)

    if similar_houses.empty: #kiểm tra tồn tại
        st.write("Lưu ý: Không có ngôi nhà phù hợp với dự đoán trong khu vực đã lựa chọn.")
    elif 'Giá (triệu đồng/m2)' in data.columns:  
        if prediction[0] >= data['Giá (triệu đồng/m2)'].mean():
            st.write("Lưu ý: Giá dự đoán ở mức cao hơn giá trung bình khu vực đã lựa chọn.")
        else:
            st.write("Lưu ý: Giá dự đoán ở mức thấp hơn giá trung bình khu vực đã lựa chọn.")
    else:
        st.write("Lưu ý: Không có dữ liệu giá trung bình khu vực đã lựa chọn.")