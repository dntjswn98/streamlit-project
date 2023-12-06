import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import plotly.io as pio # Plotly input output
import plotly.express as px # 빠르게 그리는 방법
import plotly.graph_objects as go # 디테일한 설정
import plotly.figure_factory as ff # 템플릿 불러오기
from plotly.subplots import make_subplots # subplot 만들기
from plotly.validators.scatter.marker import SymbolValidator # Symbol 꾸미기에 사용됨
import matplotlib.pyplot as plt  # 그래프 그리는 용도
import matplotlib.font_manager as fm  # 폰트 관련 용도
import seaborn as sns
import math
import streamlit as st
import datetime
import io
import plotly.figure_factory as ff

import folium
import json

from streamlit_folium import st_folium
from streamlit_folium import folium_static

st.header('서울시 공연장 관련 통계 분석 및 시각화', divider='rainbow')

# 기본 csv파일
data = pd.read_csv('./culture_space.csv', 
                   encoding="utf-8" )
data.replace('-', 0, inplace = True)
data_og = data.copy() # 원본 저장

df = data.transpose()
df.rename(columns=df.iloc[0], inplace=True)
df = df.drop(df.index[0])
df_og = df.copy()

newdf = df.reset_index()
newdf['index'] = newdf['index'].apply(lambda x : float(x))
newdf['index'] = newdf['index'].apply(lambda x : math.floor(x))

# 데이터 설명
for_prac = data_og.copy()

# 1번
연도0 = ["2013","2014","2015","2016","2017","2018","2019","2020","2021","2022"]
연도 = ["2013","2014","2015","2016","2017","2018","2019","2020","2021","2022"]
연도 = pd.DataFrame(연도)
서울 = [304, 344, 381, 389, 397, 408, 412, 417, 421, 436]
서울 = pd.DataFrame(서울)
dat = pd.concat([연도,서울], axis = 1)
dat.columns = ["연도","서울"]

# 2번, 3번 
newdf2 = newdf[(newdf['자치구']=='공공공연장') 
              | (newdf['자치구']=='민간공연장')]
newdf2 = newdf2.loc[:,["index", "자치구", "서울"]]

newdf = newdf[(newdf['자치구']=='대공연장(1000석 이상)') 
              | (newdf['자치구']=='일반공연장(300~999석)') 
              | (newdf['자치구']=='소공연장(300석 미만)')]
newdf = newdf.loc[:,['index', '자치구', '서울']]

# 4번
geojson = json.load(open('./seoulsigungu.geojson', encoding='utf-8'))
df_gu = df.loc[['2013','2014','2015','2016','2017','2018','2019','2020','2021','2022']].drop(['서울', '자치구'], axis=1)
df_gu.index.name = "연도"

#데이터가 오브젝트여서 인트로 변환
df_gut = df_gu.T
df_gut = df_gut.astype(int)

def map_change(year) :
    map = folium.Map( location = [37.541, 126.986], zoom_start=11,tiles='cartodbpositron')
    folium.Choropleth(
        geo_data = geojson,
        data = df_gut,
        columns = [df_gut.index, str(year)],
        fill_color='YlOrRd',
        #fill_opacity = 0.7,
        line_opacity=0.8,
        key_on = 'properties.SIG_KOR_NM'
    ).add_to(map)

    return map


# 사이드 바
contents_table = st.sidebar.selectbox(
    '목차를 선택해주세요',

    ('데이터 설명',
     '그래프',
     '구 별 순위'))

if contents_table == '그래프':
    option = st.sidebar.selectbox(
        '그래프를 선택해주세요',

        ('서울시 공연장 증감 추이', 
        '서울시 공연장 종류에 따른 수 차이', 
        '서울시 공연장 규모에 따른 수 차이', 
        '구 별 지도'))

# 사이드 바 확인 버튼 누르면 실행
if contents_table == '그래프':
    if option == '서울시 공연장 증감 추이':
        fig = px.line(dat, x="연도", y="서울", line_shape="linear", line_group=None, color_discrete_sequence=["blue"])
        fig.update_layout(
            title='서울시 연도별 공연장 수 추이',
            xaxis_title='연도',
            yaxis_title='공연장 수',
        )
        st.write(fig)
        st.divider()
#         explain_chart1 = """
# 서울 전체 공연장 수를 나타낸 line chart 그래프
# 매년 공연장의 수가 증가하고 있으며 2013년에서 2022년까지 10년동안 43% 증가
# 작년 공연장 수는 436개 있으며, 추가적으로 더 증가될 가능성이 높음
#         """
        st.markdown('- 서울 전체 공연장 수를 나타낸 line chart 그래프')
        st.markdown('- 매년 공연장의 수가 증가하고 있으며 2013년에서 2022년까지 10년동안 43% 증가')
        st.markdown('- 작년 공연장 수는 436개 있으며, 추가적으로 더 증가될 가능성이 높음')

    elif option == '서울시 공연장 종류에 따른 수 차이':
        fig2 =px.bar(newdf2,
                x="자치구",
                y="서울",
                facet_col='index',
                color='자치구',
                text = "서울",
                labels={'index':'y', '서울':'공연장 수', '자치구':''},
                title='서울시 공연장 종류에 따른 수 차이')
        st.write(fig2)
        st.divider()
        st.markdown('- 서울 전체 공연장 종류에 따른 공연장 수를 연도별로 나타낸 bar 그래프(공공공연장, 민간공연장)')
        st.markdown('- 2014년, 2015년은 데이터가 측정되지 않아 제외하고 그래프를 확인하였음')
        st.markdown('- 평균적으로 공공공연장의 수는 90개 , 민간공영장의 수는 300개 정도 존재함')

    elif option == '서울시 공연장 규모에 따른 수 차이':
        fig = px.bar(newdf,
              x="자치구",
              y="서울",
              title='서울시 공연장 규모에 따른 수 차이',
              hover_data=['서울'],
              color = '자치구',
             facet_col = 'index',
              labels={'index':'y', '서울':'공연장 수', '자치구':''},
             text = '서울' )
        st.write(fig)
        st.divider()
        st.markdown('- 서울 전체 공연장 규모에 따른 공연장 수를 연도별로 나타낸 bar 그래프(대, 일반, 소)')
        st.markdown('- 평균적으로 대 공연장은 20개 , 일반 공연장은 100개, 소 공연장은 280개 정도 존재함')
        st.markdown('- 대 공연장과 일반 공연장은 증감 추이가 별로 없는 반면 소 공연장은 크게 증가한 것을 볼 수 있음')
     

    elif option == '구 별 지도':
        #yyyy = st.sidebar.number_input("Insert a number", value=2013, placeholder="Type a number...")
        yyyy = st.sidebar.selectbox(
        '원하는 년도를 선택해주세요',
        (2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022))
        folium_static(map_change(yyyy))
        # https://github.com/randyzwitch/streamlit-folium
        st.divider()     
        

elif contents_table == '데이터 설명':
    st.subheader('데이터 설명')
    st.write(for_prac)
    data_explain = """
    1. "-" 로 되어있는 값은 공연장이 없는 곳이라고 생각하고 0으로 처리하여 사용함
    2. 2013년 2022년까지 서울시에 공연장 수에 대한 데이터
    3. 서울 전체의 공연장 수와 각 구의 공연장 수에 대한 데이터로 이루어져 있음
    4. 연도별로 서울시 공연장수를 공연장의 종류와 공연장의 규모로 나누어서 볼 수 있음
    5. 공연장의 종류는 공공공연장과 민간 공영장
    6. 공연장의 규모는 대(1000석이상), 일반(300~999석), 소(300석미만)
    """
    st.markdown(data_explain)
    st.divider()

    st.subheader('데이터 출처')
    st.write('https://data.seoul.go.kr/dataList/164/S/2/datasetView.do?stcSrl=164')
    st.divider()

    # 다운로드
    st.subheader('데이터 다운로드')

    @st.cache_data 
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')
    csv = convert_df(df_og)

    st.download_button(
        "Press to Download CSV",
        csv,
        "서울시_공연장_수.csv",
        "text/csv",
        key='download-csv'
    )
    st.divider()

    st.subheader('데이터 분석 목적')
    goal = """
    1. 서울시 공연장의 추가적 시각자료 제공

    \t서울문화포털 내 문화공간탭에서 볼 수 있는 서울시 내 공연장 자료는 지역에 따른 위치 표시 밖에 없어서 실질적으로 수치화된 시각자료 제공이 제한적임
    따라서 사용자의 서울시 내의 구별, 종류별, 규모별 공연장 수의 파악을 용이하도록 10개 년의 자료를 이용해 분석을 실시함

    2.  자치구별 공연 접근성 확인

    \t서울시 내의 공연장 수의 자치구별 밀집도를 분석하여 국민의 평등한 문화생활 향유 여부와 정책 개선점을 제고하기 위함"""
    st.write(goal)
    st.divider()

    st.subheader('결론')
    conclu = """
    1. 데이터를 살펴본 결과 10년 동안 종로구에 공연장 수가 가장 많은 것을 확인할 수 있음
    2. 반면 동대문구와 중랑구의 경우 10년동안 공연장 수가 매우 적은 것을 확인할 수 있음
    3.  공연장 수가 적은 구는 추가적인 지원이 필요함
    """
    st.write(conclu)

elif contents_table=='구 별 순위':
    year=st.selectbox('연도는..', 연도0)
    sorted=df.drop(['자치구', '서울'], axis=1).astype(int).sort_values(by=year, ascending=False, axis=1)
    rank=list(sorted.columns)
    st.write(f"서울 시내에서 공연장이 가장 많은 지역은 {rank[0]}입니다.")
    st.write(f"서울 시내에서 공연장이 두번째로 많은 지역은 {rank[1]}입니다.")
    st.write(f"서울 시내에서 공연장이 세번째로 많은 지역은 {rank[2]}입니다.")

else:
    st.write('좌측 사이드바에서 보고 싶은 항목을 선택해주세요 :sunglasses:')