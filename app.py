import streamlit as st
import folium
from streamlit_folium import st_folium
import osmnx as ox
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# 1. 웹 페이지 기본 설정 및 스타일 캐싱
st.set_page_config(page_title="머신러닝 보행 내비게이션", layout="wide")

st.title(" AI 기반 스마트 보행 지도")
st.markdown("### 지형 불평등과 고령층을 위한 안전 경로 예측 시스템")
st.markdown("---")

# 🤖 2. [글로벌 캐싱] 머신러닝 모델 최초 1회만 완벽 학습 (재실행 방지)
@st.cache_resource
def train_machine_learning_model():
    np.random.seed(42)
    sample_size = 300
    X_slope = np.random.uniform(0, 25, sample_size)
    X_stairs = np.where(X_slope > 10, np.random.binomial(1, 0.7, sample_size), np.random.binomial(1, 0.1, sample_size))
    X_sidewalk = np.random.randint(0, 3, sample_size)
    
    y_fatigue = (X_slope * 2.5) + (X_stairs * 20) + (X_sidewalk * 10) + np.random.normal(0, 5, sample_size)
    y_fatigue = np.clip(y_fatigue, 0, 100)
    
    df_train = pd.DataFrame({'slope': X_slope, 'stairs': X_stairs, 'sidewalk': X_sidewalk, 'fatigue_score': y_fatigue})
    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1) # 속도 최적화
    model.fit(df_train[['slope', 'stairs', 'sidewalk']], df_train['fatigue_score'])
    return model

ml_model = train_machine_learning_model()

# 🗺️ 3. [글로벌 캐싱] 네트워크 데이터 완전 고정 (클릭 시 절대 재호출 안 됨)
@st.cache_data(show_spinner=False)
def get_ml_network(region_choice):
    if "태평동" in region_choice:
        center_point = (37.4415, 127.1320)
        dist = 500 # 연산 속도를 위해 반경 최적화
        is_hilly = True
    else:
        center_point = (37.3947, 127.1112)
        dist = 500
        is_hilly = False
        
    try:
        G = ox.graph_from_point(center_point, dist=dist, network_type='walk')
        G = ox.project_graph(G, to_crs='EPSG:4326')
        
        np.random.seed(42)
        for u, v, k, data in G.edges(data=True, keys=True):
            length = data.get('length', 10)
            if is_hilly:
                slope = min(np.random.exponential(scale=7.5), 25.0)
                stairs = 1 if slope > 12 and np.random.rand() > 0.4 else 0
                sidewalk = np.random.choice([0, 1, 2], p=[0.3, 0.5, 0.2])
            else:
                slope = np.random.uniform(0.0, 3.0)
                stairs = 0
                sidewalk = np.random.choice([0, 1, 2], p=[0.8, 0.15, 0.05])
                
            data['slope'] = slope
            data['stairs'] = stairs
            data['sidewalk'] = sidewalk
            
            # AI 예측 점수 부여
            input_df = pd.DataFrame([[slope, stairs, sidewalk]], columns=['slope', 'stairs', 'sidewalk'])
            predicted_fatigue = ml_model.predict(input_df)[0]
            data['fatigue_score'] = predicted_fatigue
            data['ml_safety_weight'] = length * (1 + (predicted_fatigue / 20) ** 2)
            data['duration_min'] = length / 60.0
            
        return G
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        return None

# 4. 세션 상태 관리 (클릭 좌표 기억 장치)
if 'start_coord' not in st.session_state: st.session_state.start_coord = None
if 'end_coord' not in st.session_state: st.session_state.end_coord = None
if 'last_clicked' not in st.session_state: st.session_state.last_clicked = None

# 5. 사이드바 UI
st.sidebar.header("⚙️ 내비게이션 설정")
region_option = st.sidebar.selectbox(
    "🗺️ 대상 지역 선택", 
    ["성남시 수정구 태평동 ", "성남시 분당구 판교동 "]
)

# 지역 변경 시 자동 리셋
if 'current_region' not in st.session_state or st.session_state.current_region != region_option:
    st.session_state.start_coord = None
    st.session_state.end_coord = None
    st.session_state.last_clicked = None
    st.session_state.current_region = region_option

show_safety_layer = st.sidebar.checkbox("경사도 표시", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("🧭 내비게이션 검색 모드")
route_type = st.sidebar.radio("안내 모드 선택", ["안정 경사 경로 ", "최단 거리 경로 "])

with st.spinner("보행 도로 데이터를 구성하는 중..."):
    G = get_ml_network(region_option)

if G is None: st.stop()

st.sidebar.markdown("---")
st.sidebar.info("💡 **[클릭 방법]**\n\n지도 위를 마우스로 **첫 번째 클릭하면 출발지**, **두 번째 클릭하면 도착지**가 지정됩니다.")

# 🛠️ [핵심 추가] 사이드바 맨 아래(앱의 왼쪽 아래)에 제작자 정보를 고정합니다.
st.sidebar.markdown("<br><br><br><br><br>", unsafe_allow_html=True) # 공백을 주어 맨 아래로 밀어냄
st.sidebar.markdown("---")
st.sidebar.caption("👤 **Developed by. 김민찬 & 김성현**")

# 6. 대시보드 데이터 연산
col1, col2 = st.columns([1, 2.3])
path = None
total_distance, total_duration, max_predicted_fatigue = 0, 0, 0

if st.session_state.start_coord and st.session_state.end_coord:
    start_node = ox.nearest_nodes(G, X=st.session_state.start_coord[1], Y=st.session_state.start_coord[0])
    end_node = ox.nearest_nodes(G, X=st.session_state.end_coord[1], Y=st.session_state.end_coord[0])
    try:
        if route_type == "안정 경사 경로 (안전)":
            path = nx.shortest_path(G, source=start_node, target=end_node, weight='ml_safety_weight')
        else:
            path = nx.shortest_path(G, source=start_node, target=end_node, weight='length')
            
        fatigue_list = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            edge_data = G.get_edge_data(u, v)[0]
            total_distance += edge_data.get('length', 10)
            total_duration += edge_data.get('duration_min', 0.2)
            fatigue_list.append(edge_data.get('fatigue_score', 0))
        max_predicted_fatigue = max(fatigue_list) if fatigue_list else 0
    except:
        path = None

# 왼쪽: 결과 패널
with col1:
    st.subheader("📋 내비게이션 결과")
    if st.session_state.start_coord is None:
        st.warning("📍 지도를 클릭하여 [출발지]를 지정해 주세요.")
    elif st.session_state.end_coord is None:
        st.info("🏁 지도를 한 번 더 클릭하여 [도착지]를 지정해 주세요.")
        if st.button("🔄 검색 초기화", use_container_width=True):
            st.session_state.start_coord = None
            st.session_state.end_coord = None
            st.session_state.last_clicked = None
            st.rerun()
    else:
        st.success("✅ 경로 탐색 완료!")
        st.metric(label="⏱️ AI 예상 도보 시간", value=f"{round(total_duration)} 분")
        st.metric(label="📏 총 이동 거리", value=f"{total_distance:.0f} m")
        
        st.markdown("---")
        st.markdown("#### 🚨 AI 학습 모델 예측 위험도")
        if max_predicted_fatigue >= 65: danger_lbl, danger_cls = "🔴 고위험 ", "red"
        elif max_predicted_fatigue >= 35: danger_lbl, danger_cls = "🟠 주의 )", "orange"
        else: danger_lbl, danger_cls = "🟢 안전", "green"
            
        st.markdown(f"<div style='padding: 15px; border-left: 5px solid {danger_cls}; background-color: #f9f9f9; font-weight: bold;'>{danger_lbl}</div>", unsafe_allow_html=True)
        st.markdown(f"*보행 피로도 예측 : `{max_predicted_fatigue:.1f} / 100점`*")
        
        st.markdown("---")
        if st.button("🔄 검색 초기화", use_container_width=True):
            st.session_state.start_coord = None
            st.session_state.end_coord = None
            st.session_state.last_clicked = None
            st.rerun()

    st.markdown("---")
    st.markdown("#### 📊 지역 보행 인프라 분석")
    all_slopes = [d['slope'] for u, v, k, d in G.edges(data=True, keys=True)]
    st.text(f"• 해당 구역 평균 경사: {np.mean(all_slopes):.1f}°")
    st.text(f"• 12° 이상 급경사 도로 비중: {(sum(1 for s in all_slopes if s>=12)/len(all_slopes))*100:.1f}%")

# 오른쪽: 지도
with col2:
    st.subheader(" 내비게이션 맵")
    nodes_dict = list(G.nodes(data=True))
    init_lat, init_lng = nodes_dict[0][1]['y'], nodes_dict[0][1]['x']
    
    m = folium.Map(location=[init_lat, init_lng], zoom_start=16, tiles="CartoDB positron")
    
    for u, v, k, data in G.edges(data=True, keys=True):
        geo_points = [(G.nodes[u]['y'], G.nodes[u]['x']), (G.nodes[v]['y'], G.nodes[v]['x'])]
        if show_safety_layer:
            slope = data['slope']
            color = '#FF4B4B' if slope >= 12 else ('#FFA500' if slope >= 5 else '#2EA043')
            weight = 2.5
        else:
            color = '#E0E0E0'
            weight = 1.5
        folium.PolyLine(geo_points, color=color, weight=weight, opacity=0.6).add_to(m)
        
    if st.session_state.start_coord:
        folium.Marker(st.session_state.start_coord, popup="출발지", icon=folium.Icon(color='blue', icon='play')).add_to(m)
    if st.session_state.end_coord:
        folium.Marker(st.session_state.end_coord, popup="도착지", icon=folium.Icon(color='red', icon='flag')).add_to(m)
    if path:
        path_coords = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in path]
        folium.PolyLine(path_coords, color='#0047FF', weight=6, opacity=0.95).add_to(m)
        
    map_data = st_folium(m, width=950, height=600, key="nav_map_fixed")
    
    if map_data and map_data.get("last_clicked"):
        click_event = map_data["last_clicked"]
        current_click = (click_event["lat"], click_event["lng"])
        
        if st.session_state.last_clicked != current_click:
            st.session_state.last_clicked = current_click
            
            if st.session_state.start_coord is None:
                st.session_state.start_coord = current_click
                st.rerun()
            elif st.session_state.end_coord is None and current_click != st.session_state.start_coord:
                st.session_state.end_coord = current_click
                st.rerun()
