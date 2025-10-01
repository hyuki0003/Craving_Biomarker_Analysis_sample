import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import neurokit2 as nk
import re
import warnings
from datetime import datetime, timedelta
from collections import Counter
from math import sqrt
from scipy.signal import lombscargle
from neurokit2.misc import NeuroKitWarning
warnings.filterwarnings(action='ignore')
warnings.filterwarnings("ignore", category=NeuroKitWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
from warnings import warn
from neurokit2 import signal_autocor
from neurokit2.eda import eda_sympathetic

FS_ECG = 512
FS_PPG = 51.2

def data_split(base_dir: str, output_dir: str, check_ctl_inter=False):
    """
    주어진 한 Subject의 원본 데이터를 분석하여 VR 타임스탬프를 기준으로 ECG, PPG/GSR 데이터를 'low', 'mid', 'high' 등의 구간으로 분할하고 (output_dir)에 저장.

    Args:
        base_dir (str): 'ECG_PPG_GSR', 'VR_Timestamp' 폴더를 포함하는 한 Subject 원본 데이터의 최상위 경로.
        output_dir (str): 분할된 CSV 파일들을 저장할 목적지 경로.
        check_list (str): control과 intervene feature도 뽑을 것인지 확인
    """
    # ------------------------- 1. 데이터 경로 변수 초기화 -------------------------
    # PPG, GSR 파일 경로
    PPG_path = ''
    # ECG 파일 경로
    ECG_path = ''
    # VR Timestamp 파일 경로
    VR_timestamp_path = ''

    print(base_dir+ ' 데이터 Split 처리 시작')

    # ------------------------- 2. Subject 내부를 순회하며 데이터 경로 탐색 -------------------------

    # subject 폴더 내부에 ECG_PPG_GSR이라는 폴더가 있는지 확인
    if 'ECG_PPG_GSR' in os.listdir(base_dir):
        # 'ECG_PPG_GSR' 폴더 내부의 하위 폴더들 순회
        for path in os.listdir(base_dir + '/ECG_PPG_GSR'):
            temp_path = os.path.join(base_dir, 'ECG_PPG_GSR', path)
            ecg_ppg_path = os.listdir(temp_path)[0]
            # 파일명에 포함된 장비 ID로 PPG, ECG 경로 특정
            if 'id95AE' in ecg_ppg_path:
                PPG_path = temp_path + '/' + ecg_ppg_path
            elif 'Shimmer_820D' in ecg_ppg_path:
                ECG_path = temp_path + '/' + ecg_ppg_path

    # subject 폴더 내부 VR_Timestamp 폴더 확인
    if 'VR_Timestamp' in os.listdir(base_dir):
        VR_timestamp_path = base_dir + '/VR_Timestamp/' + os.listdir(base_dir + '/VR_Timestamp')[0]
    # ------------------------- 3. 필수 파일 경로 확인 -------------------------
    # 세 개의 경로 중 하나라도 비어있으면 오류 메시지 출력 후 함수 종료
    if PPG_path == '':
        print('폴더 내 PPG 파일이 없습니다.')
        return
    elif ECG_path == '':
        print('폴더 내 ECG 파일이 없습니다.')
        return
    elif VR_timestamp_path == '':
        print('폴더 내 VR_timestamp 파일이 없습니다.')
        return

    # 각 센서 데이터의 타임스탬프 컬럼명 정의
    PPG_timestamp_column_name = 'id95AE_Timestamp_Unix_CAL'
    ECG_timestamp_column_name = 'Shimmer_820D_Timestamp_Unix_CAL'

    # ------------------------- 4. 센서 데이터 및 VR 타임스탬프 로드 -------------------------

    # 센서 데이터 로드
    PPG_df = load_csv(PPG_path)
    ECG_df = load_csv(ECG_path)

    # VR 타임스탬프 파일에서 각 구간(start, end, low, mid, high)의 시간 정보 추출
    start, end, low, mid, high, control, intervene = get_VR_timestamp(file_path=VR_timestamp_path, temp=check_ctl_inter)

    # ------------------------- 5. VR 타임스탬프 기준으로 데이터 필터링 (PPG & ECG) -------------------------
    # PPG 데이터를 각 구간별로 필터링
    start_PPG = filter_data_by_time(PPG_df, start, PPG_timestamp_column_name)
    end_PPG = filter_data_by_time(PPG_df, end, PPG_timestamp_column_name)
    low_PPG = filter_data_by_time(PPG_df, low, PPG_timestamp_column_name)
    mid_PPG = filter_data_by_time(PPG_df, mid, PPG_timestamp_column_name)
    high_PPG = filter_data_by_time(PPG_df, high, PPG_timestamp_column_name)

    # ECG 데이터를 각 구간별로 필터링
    start_ECG = filter_data_by_time(ECG_df, start, ECG_timestamp_column_name)
    end_ECG = filter_data_by_time(ECG_df, end, ECG_timestamp_column_name)
    low_ECG = filter_data_by_time(ECG_df, low, ECG_timestamp_column_name)
    mid_ECG = filter_data_by_time(ECG_df, mid, ECG_timestamp_column_name)
    high_ECG = filter_data_by_time(ECG_df, high, ECG_timestamp_column_name)

    # ------------------------- 6. 분할된 데이터를 CSV 파일로 저장 -------------------------

    # 데이터를 Label에 맞게 저장
    save_filtered_data(start_PPG, output_dir, "start", "PPG")
    save_filtered_data(end_PPG, output_dir, "end", "PPG")
    save_filtered_data(start_ECG, output_dir, "start", "ECG")
    save_filtered_data(end_ECG, output_dir, "end", "ECG")

    save_filtered_data(low_PPG, output_dir, "low", "PPG")
    save_filtered_data(mid_PPG, output_dir, "mid", "PPG")
    save_filtered_data(high_PPG, output_dir, "high", "PPG")

    save_filtered_data(low_ECG, output_dir, "low", "ECG")
    save_filtered_data(mid_ECG, output_dir, "mid", "ECG")
    save_filtered_data(high_ECG, output_dir, "high", "ECG")

    if control != None and intervene != None and check_ctl_inter:

        control_PPG = filter_data_by_time(PPG_df, control, PPG_timestamp_column_name)
        control_ECG = filter_data_by_time(ECG_df, control, ECG_timestamp_column_name)

        intervene_PPG = filter_data_by_time(PPG_df, intervene, PPG_timestamp_column_name)
        intervene_ECG = filter_data_by_time(ECG_df, intervene, ECG_timestamp_column_name)

        save_filtered_data(control_PPG, output_dir, "control", "PPG")
        save_filtered_data(intervene_PPG, output_dir, "intervene", "PPG")
        save_filtered_data(control_ECG, output_dir, "control", "ECG")
        save_filtered_data(intervene_ECG, output_dir, "intervene", "ECG")

    print(output_dir+'폴더에 Split데이터를 저장했습니다.\n')

def load_csv(file_path):
    """
    Shimmer 센서 데이터 형식의 CSV 파일을 로드.

    Args:
        file_path (str): 로드할 CSV 파일의 경로.

    Returns:
        pd.DataFrame: 로드된 데이터가 담긴 DataFrame.
    """

    return pd.read_csv(file_path, skiprows=[0, 2], sep='\t', low_memory=False)

def get_VR_timestamp(file_path, temp=False):
    """
    VR_timestamp 파일을 읽어서 낮은 갈망(low), 중간 갈망(mid), 높은 갈망(high) 유발 영상의 시작과 종료 시간을 받는다.
    :param file_path: VR_timestamp의 file 경로로.
    :return: low(낮은 갈망), mid(중간 갈망), high(높은 갈망) 각각의 타임스탬프가 저장된 list.
    """
    df = pd.read_excel(file_path)

    def find_index_by_keyword(column_index, keyword):
        """
        특정 열에서 특정 키워드가 포함된 행의 인덱스를 찾는 함수.
        :param column_index: 검색할 열의 인덱스
        :param keyword: 찾을 키워드 (예: '_낮은_', '_중간_', '_높은_')
        :return: 해당 키워드를 포함하는 행의 인덱스 리스트
        """
        return df[df.iloc[:, column_index].astype(str).str.contains(keyword, na=False)].index.tolist()

    start_end_range = sorted(find_index_by_keyword(column_index=2, keyword='괌'))
    if len(start_end_range) >= 3:
        control_range = sorted(find_index_by_keyword(column_index=2, keyword='통제'))
        intervene_range = sorted(find_index_by_keyword(column_index=2, keyword='중재'))
        ctr = find_index_by_keyword(column_index=2, keyword='통제')
        inv = find_index_by_keyword(column_index=2, keyword='중재')
        ctr_sum = sum(ctr)
        inv_sum = sum(inv)
        if ctr_sum > inv_sum:
            check_first = '중재'
        else:
            check_first = '통제'
        control_gwam_range, intervene_gwam_range = control_range[1], intervene_range[1]

    start_range, end_range = start_end_range[0], start_end_range[1]

    low_range = find_index_by_keyword(column_index=2, keyword='_낮은_')
    mid_range = find_index_by_keyword(column_index=2, keyword='_중간_')
    high_range = find_index_by_keyword(column_index=2, keyword='_높은_')



    def extract_timestamps(rng):
        result = []
        for i in rng:
            try:
                # 종료 시각 파싱
                end_time_raw = df.iloc[i, 9]

                if 'PM' in end_time_raw or 'AM' in end_time_raw:
                    # 12시간제 처리
                    end_time = datetime.strptime(end_time_raw.strip(), "%Y-%m-%d %I:%M:%S.%f %p")
                else:
                    # 24시간제 처리
                    end_time_str = end_time_raw[:-3]
                    end_time = datetime.strptime(end_time_str, "%Y-%m-%d %H:%M:%S.%f")

                # 재생시간 파싱
                mins, secs = df.iloc[i, 10].split(":")
                delta = timedelta(minutes=int(mins), seconds=float(secs))

                # 원래 start_time 계산
                start_time = end_time - delta

                # print('before start : ', datetime.strftime(start_time, "%Y-%m-%d %H:%M:%S.%f")[:-3])
                # print('before end   : ', datetime.strftime(end_time, "%Y-%m-%d %H:%M:%S.%f")[:-3])

                # 마진 1초 적용 (start+1초, end-1초)
                margin = timedelta(seconds=1)
                start_time_margin = start_time + margin
                end_time_margin = end_time - margin

                # print('after  start : ', datetime.strftime(start_time_margin, "%Y-%m-%d %H:%M:%S.%f")[:-3])
                # print('after  end   : ', datetime.strftime(end_time_margin, "%Y-%m-%d %H:%M:%S.%f")[:-3])

                # 유효성 체크
                if start_time_margin >= end_time_margin:
                    print(f"[⚠️] index {i}: 마진 적용 후 시작 시각이 종료 시각과 같거나 이후입니다. 무시됩니다.")
                    continue

                # 문자열로 변환
                start_str = datetime.strftime(start_time_margin, "%Y-%m-%d %H:%M:%S.%f")[:-3]
                end_str = datetime.strftime(end_time_margin, "%Y-%m-%d %H:%M:%S.%f")[:-3]

                result.append((start_str, end_str))

            except Exception as e:
                print(f"index {i} 처리 중 오류: {e}")

        if not result:
            print("지정된 범위 내 파싱 가능한 데이터가 없습니다.")

        return result

    start = extract_timestamps([start_range])
    end = extract_timestamps([end_range])

    low = extract_timestamps(low_range)
    mid = extract_timestamps(mid_range)
    high = extract_timestamps(high_range)
    if len(start_end_range) >= 3 and temp:
        control = extract_timestamps([control_gwam_range])
        intervene = extract_timestamps([intervene_gwam_range])
        return start, end, low, mid, high, control, intervene
    return start, end, low, mid, high, None, None

def convert_to_unix(time_str):
    """
        'YYYY-MM-DD HH:MM:SS.sss' 형식의 시간 문자열을 UNIX 타임스탬프로 변환.

        Args:
            time_str (str): 변환할 시간 문자열.

        Returns:
            int | None: 변환된 UNIX 타임스탬프 또는 변환 실패 시 None.
    """
    try:
        # UNIX 타임스탬프로 변환 시도
        dt = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S.%f')
        #
        return int(dt.timestamp() * 1000)
    except ValueError:
        print('Error')
        return None

def filter_data_by_time(df, timestamps, timestamp_column_name = 'Timestamp'):
    """
        주어진 (시작, 종료) 시각 리스트에 따라 DataFrame을 필터링.

        Args:
            df (pd.DataFrame): 필터링할 원본 DataFrame.
            timestamps (list[tuple[str, str]]): (시작 시각, 종료 시각) 문자열 튜플의 리스트.
            timestamp_column_name (str): df에 저장되어있는 UNIX 타임스탬프의 컬럼 명.

        Returns:
            list[pd.DataFrame]: 각 타임스탬프 구간에 맞게 필터링된 DataFrame 리스트.
    """
    filtered_data = []
    # 각 (시작, 종료) 쌍에 대해 반복
    for start, end in timestamps:
        # 시간 문자열을 UNIX 타임스탬프로 변환
        start_time = convert_to_unix(time_str = start)
        end_time = convert_to_unix(time_str = end)
        # 시작 시간과 종료 시간 사이의 데이터만 필터링하여 리스트에 추가
        filtered_data.append(df[(df[timestamp_column_name] >= start_time) & (df[timestamp_column_name] <= end_time)])
    return filtered_data

def save_filtered_data(filtered_data, base_dir, category, prefix):
    """
        필터링된 데이터프레임 리스트를 지정된 경로에 CSV 파일로 저장.

        Args:
            filtered_data (list[pd.DataFrame]): 저장할 DataFrame의 리스트.
            base_dir (str): 저장할 최상위 경로 (e.g., output_dir).
            category (str): 하위 폴더 이름 및 파일명 (e.g., 'low', 'mid').
            prefix (str): 추출한 데이터의 이름 (e.g., 'PPG', 'ECG').
    """
    # 저장 경로 생성 (e.g., output_dir/PPG/low)
    dir_path = os.path.join(base_dir, prefix, category)
    os.makedirs(dir_path, exist_ok=True)
    # 리스트 내 각 데이터프레임을 별도 파일로 저장 (e.g., low1.csv, low2.csv, ...)
    for idx, df in enumerate(filtered_data, start=1):
        df.to_csv(os.path.join(dir_path, f"{category}{idx}.csv"), index=False)

def get_label(base_dir: str, sam_result_path: str, subject_path: str, output_dir: str):
    """
    한 Subject의 음주 갈망 설문(SAM) 결과를 Excel 파일에서 찾아 처리하고, 결과를 파일별 라벨이 담긴 CSV로 저장.

    Args:
        base_dir (str): Subject의 raw_data 경로 VR_Timestamp를 확보하기 위함.
        sam_result_path (str): 강동성심병원(KD.xlsx)와 춘천성심병원(CC.xlsx) 설문 결과 파일이 들어있는 디렉토리 경로.
        subject_path (str): 처리할 Subject의 이름 또는 ID.
        output_dir (str): 최종 라벨 CSV 파일을 저장할 디렉토리 경로.

    Returns:
        pd.dataframe | None: 한 Subject의 라벨값이 들어있는 dataframe 오류가 발생하면 None 값 전송
    """

    # 경로 및 라벨 변수 초기화
    VR_timestamp_path = ''
    trial_label_more = None
    trial_label_over = None
    trial_label_avg = None
    Q1 = None
    Q2 = None

    print(base_dir+' Label 정보 처리 시작')

    # 필수 데이터 파일 경로 탐색 (VT_Timestamp)
    if 'VR_Timestamp' in os.listdir(base_dir):
        VR_timestamp_path = base_dir + '/VR_Timestamp/' + os.listdir(base_dir + '/VR_Timestamp')[0]
    elif VR_timestamp_path == '':
        print('폴더 내 VR_timestamp 파일이 없습니다.')
        return

    # Subject 정보 파싱 및 라벨 파일 로드
    subject_split = subject_path.split('_')
    subject_path = subject_path.split('(')
    subject_path = subject_path[0]

    # 대조군과 알콜환자 및 병원 분류 후 라벨 데이터 로드
    # 알코올 그룹
    if subject_split[0] == '1':
        # 강동성심병원
        if subject_split[1] == '1':
            data = pd.read_excel(sam_result_path+r'\KD.xlsx', sheet_name='알코올음주갈망')
            data.drop(21,inplace=True)
            trial_label_more, trial_label_over, trial_label_avg, Q1, Q2 = subject_SAM_result(data, subject_path, subject_check=0)
        # 춘천성심병원
        elif subject_split[1] == '2':
            data = pd.read_excel(sam_result_path+r'\CC.xlsx', sheet_name='알코올음주갈망')
            data = data[2:]
            trial_label_more, trial_label_over, trial_label_avg, Q1, Q2 = subject_SAM_result(data, subject_path, subject_check=0)
    # 대조군 그룹
    if subject_split[0] == '3':
        # 강동성심병원
        if subject_split[1] == '1':
            data = pd.read_excel(sam_result_path+r'\KD.xlsx', sheet_name='대조군음주갈망')
            trial_label_more, trial_label_over, trial_label_avg, Q1, Q2 = subject_SAM_result(data, subject_path, subject_check=1)
        # 춘천성심병원
        if subject_split[1] == '2':
            data = pd.read_excel(sam_result_path+r'\CC.xlsx', sheet_name='대조군음주갈망')
            trial_label_more, trial_label_over, trial_label_avg, Q1, Q2 = subject_SAM_result(data, subject_path, subject_check=1)

    # 라벨 데이터 유효성 검사
    if trial_label_over == None or trial_label_more == None or trial_label_avg == None or Q1 == None or Q2 == None:
        print(subject_path + '의 라벨값이 없습니다.')
        return None

    # VR 타임스탬프에서 low, mid, high 영상 개수 확인
    start, end, low, mid, high, _, _ = get_VR_timestamp(file_path=VR_timestamp_path)

    # 영상 파일명 리스트 생성
    label = []
    for i in range(len(low)):
        label.append('low'+str(i+1)+'.csv')
    for i in range(len(mid)):
        label.append('mid'+str(i+1)+'.csv')
    for i in range(len(high)):
        label.append('high'+str(i+1)+'.csv')
    # 영상 파일 개수와 추출된 라벨 개수가 일치하는지 확인
    if len(label) != len(trial_label_over) or len(label) != len(trial_label_more) or len(label) != len(trial_label_avg) or len(label) != len(Q1) or len(label) != len(Q2):
        print(subject_path+'에 누락된 라벨이 있습니다.')
        return None

    # 최종 데이터프레임 생성 및 저장
    df = pd.DataFrame()

    df["File"] = label
    # 3.5점 이상이면 1
    df["label-3.5_more"] = trial_label_more
    # 3.5점 초과이면 1
    df["label-3.5_over"] = trial_label_over
    # Q1, Q2 평균값
    df["label-avg"] = trial_label_avg
    # Q1 값
    df['Q1'] = Q1
    # Q2 값
    df['Q2'] = Q2

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_file = os.path.join(output_dir, subject_path + '.csv')
        df.to_csv(save_file, index=False)
        print(f"{save_file}에 Label 정보를 저장했습니다.\n")
    return df

def subject_SAM_result(data, subject_name, subject_check=0):
    """
    DataFrame에서 특정 Subject의 SAM 설문 결과를 찾아 trial별로 처리.

    Args:
        data (pd.DataFrame): 설문 결과가 담긴 DataFrame (Excel 시트에서 로드).
        subject_name (str): 찾을 Subject의 이름.
        subject_check (int, optional): 그룹을 구분하는 플래그. 0: 알코올 그룹, 1: 대조군.

    Returns:
        list:
            처리된 라벨 리스트.
            (subject_more, subject_over, subject_avg, subject_Q1, subject_Q2)
            Subject를 찾지 못하거나 오류 발생 시 (None, None, None, None, None) 반환.
    """
    # 데이터 전처리 (필요 없는 정보 제거)
    data.drop(axis=1, labels=['참여자 ID', 'Visit 세션'], inplace=True)
    data.reset_index(drop=True, inplace=True)

    # 결과를 저장할 리스트 초기화
    subject_avg = []
    subject_more = []
    subject_over = []
    subject_Q1 = []
    subject_Q2 = []

    # 그룹(알코올/대조군)에 따라 설문 점수가 시작되는 컬럼 인덱스 설정
    if subject_check == 0:
        q1 = 2
        q2 = 25
    else:
        q1 = 2
        q2 = 20

    # DataFrame을 순회하며 일치하는 Subject 탐색
    for i in range(len(data)):
        temp = data.iloc[i]
        if temp[0] == subject_name:
            if np.isnan(temp[2]) == False:
                # Subject를 찾으면, trial별 점수 추출
                for i in range(18):
                    # 데이터가 '-'와 같은 비숫자 값이면 중단
                    if temp[i+q1] == '-' or temp[i+q2] == '-':
                        break
                    try:

                        # 5. 라벨 계산 및 리스트에 추가
                        subject_avg.append((int(temp[i+q1]) + int(temp[i+q2])) / 2)
                        subject_more.append(1 if((int(temp[i+q1]) + int(temp[i+q2])) / 2) >= 3.5 else 0)
                        subject_over.append(1 if((int(temp[i+q1]) + int(temp[i+q2])) / 2) > 3.5 else 0)
                        subject_Q1.append(int(temp[i+q1]))
                        subject_Q2.append(int(temp[i+q2]))
                    except:
                        # 변환 중 오류 발생 시 None 반환
                        return None, None, None, None, None
                # Subject의 모든 trial 처리가 끝나면 결과 반환
                return subject_more, subject_over, subject_avg, subject_Q1, subject_Q2
     # 6. DataFrame 전체를 순회해도 Subject를 찾지 못한 경우
    return None, None, None, None, None

def extract_signal(df, data_type = 'ECG'):
    """
    데이터프레임에서 생체 신호 추출.

    Args:
        df (pd.DataFrame): Shimmer 센서 데이터가 포함된 원본 데이터프레임.
        data_type (str): 추출할 신호의 종류. 'ECG', 'PPG', 'GSR' 중 하나를 선택.

    Returns:
        dict: 채널 이름을 키(key)로, 해당 신호(signal)를 값(value)으로 갖는 딕셔너리.
              만약 data_type이 유효하지 않으면 빈 딕셔너리를 반환.
    """

    columns = {
        'ECG': {
            'LA_RA' : 'Shimmer_820D_ECG_LA-RA_24BIT_CAL',
            'LL_LA' : 'Shimmer_820D_ECG_LL-LA_24BIT_CAL',
            'LL_RA' : 'Shimmer_820D_ECG_LL-RA_24BIT_CAL',
            'Vx_RL' : 'Shimmer_820D_ECG_Vx-RL_24BIT_CAL'
        },
        'PPG' : {
            'ppg' : 'id95AE_PPG_A13_CAL'
        },
        'GSR' : {
            'gsr' : 'id95AE_GSR_Skin_Conductance_CAL'

        }
    }

    # df에서 신호를 추출
    signal = {key: df[col].values for key, col in columns[data_type].items()}
    return signal

def ECG_metrics(subject_ECG_dir_path, save_path=None, show=False, unit=None):
    """
    한 Subject의 모든 ECG 데이터를 분석하여 HR, HRV 지표를 계산하고 결과를 반환.


    Args:
        subject_ECG_dir_path (str): 분석할 Subject의 ECG 데이터가 담긴 상위 디렉토리 경로. (e.g: '.../data/sub1/ECG')
        save_path (str, optional): 분석 결과(CSV, plot)를 저장할 최상위 디렉토리 경로.
        show (bool, optional): 분석 과정에서 생성되는 plot을 화면에 표시할지 여부. save_path가 지정된 경우에만 활성화.
        unit (str, optional): ECG 신호의 단위 (예: 'mV'). Plot 제목에 사용.

    Returns:
        pd.DataFrame:
            Subject의 모든 파일과 채널에 대한 HR, HRV 분석 결과가 포함된 데이터프레임.
    """
    print(subject_ECG_dir_path+ ' ECG Signal - HR, HRV 분석 시작')
    # 최종 결과를 저장하기 위한 빈 데이터프레임 생성
    df = pd.DataFrame()

    # subject 내 ECG 디렉토리 파일들 순회 (low, mid, high)
    for dir_name in os.listdir(subject_ECG_dir_path):
        dir_path = os.path.join(subject_ECG_dir_path, dir_name)

        # 데이터 저장을 위한 Subject 이름 추출
        subject_path = subject_ECG_dir_path.split('\\')[3]
        # 시각화를 원한다면 시각화 결과를 저장할 폴더 생성
        if save_path and show:
            os.makedirs(save_path + '/' + subject_path, exist_ok=True)

        # ECG 데이터 파일 순회 (low1.csv, low2.csv, ...)
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)

            # 파일명에서 확장자를 제외한 부분 추출 (e.g., 'low1')
            title_name = file_name.split('.')[0]

            # CSV 파일에서 ECG 신호 추출
            raw_signal = extract_signal(pd.read_csv(file_path), 'ECG')

            # 한 파일의 분석 결과를 임시로 담을 데이터프레임 생성
            temp = pd.DataFrame()
            temp["File"] = [file_name]

            # 추출된 신호의 각 채널별로 처리
            for channel_name, raw in raw_signal.items():
                temp["channel"] = channel_name

                # Plot에 사용할 제목 생성
                title = '' + title_name + ' - ECG ' + channel_name
                # HR(심박수) 계산 및 Plot 저장
                HR = save_ECG_HR_plot(raw, show, title=title,
                                      save_path=save_path + '/' + subject_path + '/' + title_name + ' - ECG ' + channel_name if save_path and show else None,
                                      unit=unit)
                temp["HR"] = HR

                # HRV(심박 변이도) 지표 계산 및 Plot 저장
                HRV = save_ECG_HRV_plot(raw, show, title=title,
                                        save_path=save_path + '/' + subject_path + '/' + title_name + ' - ECG ' + channel_name if save_path and show else None)

                # HR과 HRV 결과를 임시 데이터프레임에 병합
                result = pd.concat([temp, HRV], axis=1)

                # 최종 데이터프레임에 현재 채널의 결과 누적
                if df.size == 0:
                    df = result
                else:
                    df = pd.concat([df, result], ignore_index=True)

    # 최종 결과를 CSV 파일로 저장 (save_path가 지정된 경우)
    if save_path:
        # 저장 경로가 없으면 생성
        os.makedirs(save_path, exist_ok=True)
        # CSV로 저장
        save_file = os.path.join(save_path, subject_ECG_dir_path.split('\\')[-2] + '.csv')
        df.to_csv(save_file, index=False)

        # 파일 저장 완료 메시지 출력
        print(f"{save_file}에 ECG Signal - HR, HRV 분석 결과를 저장했습니다.")
    # 분석 결과가 담긴 최종 데이터프레임 반환
    return df

def save_ECG_HR_plot(ecg_signal, show=False, title=None, save_path=None, unit=None):
    """
        ECG 신호를 분석하여 평균 심박수(HR)를 계산하고, 관련 Plot 저장.

        Args:
            ecg_signal (np.ndarray): 분석할 Raw ECG 신호.
            show (bool, optional): 분석 Plot을 화면에 표시할지 여부.
            title (str, optional): Plot에 표시될 제목.
            save_path (str, optional): Plot을 저장할 경로 및 파일명.
            unit (str, optional): 신호의 단위 (y축 라벨에 사용).

        Returns:
            float: 계산된 평균 심박수(ECG_Rate_Mean). 오류 발생 시 0을 반환.
    """
    try:
        # ECG 전처리 및 분석
        signals, info = nk.ecg_process(ecg_signal, sampling_rate=FS_ECG)
        analyze_df = nk.ecg_analyze(signals, sampling_rate=FS_ECG)

        if save_path:
            nk.ecg_plot(signals, info, title + ' (HR)', save_path + ' (HR)', unit, show=show)
        # 시각화 옵션 (show = True)

        # 평균 심박수 추출
        try:
            mean_hr = analyze_df['ECG_Rate_Mean'].values[0]
            # print(f"Calculated HR: {mean_hr}")  # HR 값을 출력
        except KeyError:
            print("Error: 'ECG_Rate_Mean' not found in analysis.")
            mean_hr = None
    except Exception as e:
        print(e)
        mean_hr = 0
    return mean_hr

def save_ECG_HRV_plot(ecg_signal, show=False, title=None, save_path=None):
    """
        ECG 신호에서 심박 변이도(HRV) 지표들을 계산하고, 관련 Plot 저장.

        Args:
            ecg_signal (np.ndarray): 분석할 Raw ECG 신호.
            show (bool, optional): 분석 Plot을 화면에 표시할지 여부.
            title (str, optional): Plot에 표시될 제목.
            save_path (str, optional): Plot을 저장할 경로 및 파일명.

        Returns:
            pd.DataFrame | None:
                계산된 모든 HRV 지표가 포함된 DataFrame. 오류 발생 시 None을 반환.
    """
    try:
        # print("Signal length:", len(ecg_signal))

        # ECG 전처리 및 분석 peak 추출
        signals, info = nk.ecg_process(ecg_signal, sampling_rate=FS_ECG)
        peaks, info = nk.ecg_peaks(signals, sampling_rate=FS_ECG)

        # 검출된 peak를 기반으로 HRV 지표 계산 및 시각화
        hrv = nk.hrv(peaks, sampling_rate=FS_ECG, show=show, title=title + ' (HRV)', save_path=save_path + ' (HRV)' if save_path else None)
    except Exception as e:
        print(e)
        return None
    return hrv

def PPG_metrics(subject_PPG_dir_path, save_path=None, show=False, unit=None):
    """
    한 Subject의 모든 PPG 파일을 분석하여 HR, HRV 지표를 계산하고 결과를 반환.

    Args:
        subject_PPG_dir_path (str): 분석할 Subject의 PPG 데이터가 담긴 디렉토리 경로.(예: '.../data/sub1/PPG')
        save_path (str, optional): 분석 결과(CSV, plot)를 저장할 최상위 디렉토리 경로.
        show (bool, optional): 분석 과정에서 생성되는 plot을 화면에 표시할지 여부. save_path가 지정된 경우에만 활성화.
        unit (str, optional): PPG 신호의 단위. Plot에 사용.

    Returns:
        pd.DataFrame:
            Subject의 모든 PPG 파일에 대한 HR, HRV 분석 결과가 포함된 데이터프레임.
    """
    print(subject_PPG_dir_path+ ' PPG Signal - HR, HRV 분석 시작')
    # 최종 결과를 저장하기 위한 빈 데이터프레임 생성
    df = pd.DataFrame()

    # subject 내 PPG 디렉토리 파일들 순회 (low, mid, high)
    for dir_name in os.listdir(subject_PPG_dir_path):
        dir_path = os.path.join(subject_PPG_dir_path, dir_name)

        # 데이터 저장을 위한 Subject 이름 추출
        subject_path = subject_PPG_dir_path.split('\\')[3]
        # 시각화를 원한다면 시각화 결과를 저장할 폴더 생성
        if save_path and show:
            os.makedirs(save_path + '/' + subject_path, exist_ok=True)

        # PPG 데이터 파일 순회 (low1.csv, low2.csv, ...)
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)

            # 파일명에서 확장자를 제외한 부분 추출 (e.g., 'low1')
            title_name = file_name.split('.')[0]
            try:

                # CSV 파일에서 PPG 신호 추출
                raw_df = pd.read_csv(file_path)
                raw_signal = extract_signal(raw_df, 'PPG')
                raw = raw_signal['ppg']

                # 한 파일의 분석 결과를 임시로 담을 데이터프레임 생성
                temp = pd.DataFrame()
                temp["File"] = [file_name]
                temp["channel"] = "ppg"

                title = '' + title_name + ' - PPG '

                # HR(심박수) 계산 및 Plot 저장
                HR = save_PPG_HR_plot(raw, show, title=title,
                                      save_path=save_path + '/' + subject_path + '/' + title_name + ' - PPG ' if save_path and show else None,
                                      unit=unit)
                temp["HR"] = HR

                # HRV(심박 변이률) 계산 및 Plot 저장
                HRV = save_PPG_HRV_plot(raw, show, title=title,
                                        save_path=save_path + '/' + subject_path + '/' + title_name + ' - PPG ' if save_path and show else None)

                # HR과 HRV 결과를 임시 데이터프레임에 병합
                result = pd.concat([temp, HRV], axis=1)

                # 최종 데이터프레임에 현재 채널의 결과 누적
                df = pd.concat([df, result], ignore_index=True)

            except Exception as e:
                print(f"Failed to process {file_path}: {e}")
                import traceback
                traceback.print_exc()
    # 최종 결과를 CSV 파일로 저장 (save_path가 지정된 경우)
    if save_path:
        # 저장 경로가 없으면 생성
        os.makedirs(save_path, exist_ok=True)
        # CSV로 저장
        save_file = os.path.join(save_path, subject_PPG_dir_path.split('\\')[-2] + '.csv')
        df.to_csv(save_file, index=False)

        # 파일 저장 완료 메시지 출력
        print(f"{save_file}에 PPG Signal - HR, HRV 분석 결과를 저장했습니다.")
    # 분석 결과가 담긴 최종 데이터프레임 반환
    return df

def save_PPG_HR_plot(ppg_signal, show=False, title=None, save_path=None, unit=None):
    """
    PPG 신호를 분석하여 평균 심박수(HR)를 계산하고, 관련 Plot을 저장.

    Args:
        ppg_signal (np.ndarray): 분석할 Raw PPG 신호 배열.
        show (bool, optional): 분석 Plot을 화면에 표시할지 여부.
        title (str, optional): Plot에 표시될 제목.
        save_path (str, optional): Plot을 저장할 경로 및 파일명.
        unit (str, optional): 신호의 단위 (y축 라벨에 사용).

    Returns:
        float | None: 계산된 평균 심박수(PPG_Rate_Mean). 오류 발생 시 None을 반환.
    """
    try:
        # PPG 신호 처리 및 분석
        signals, info = nk.ppg_process(ppg_signal, sampling_rate=FS_PPG)
        analyze_df = nk.ppg_analyze(signals, sampling_rate=FS_PPG)

        # save_path가 제공되면 PPG 처리 과정 Plot 저장
        if save_path:
            nk.ppg_plot(signals, info, show=show, title=title + ' (HR)', save_path=save_path + ' (HR)', unit=unit)

        # 분석 결과에서 평균 심박수(HR) 값 추출
        mean_hr = analyze_df['PPG_Rate_Mean'].values[0]

    except Exception as e:
        print(f"[ERROR - get_PPG_HR] {e}")
        mean_hr = None

    return mean_hr


def save_PPG_HRV_plot(ppg_signal, show=False, title=None, save_path=None):
    """
    PPG 신호에서 심박 변이도(HRV) 지표들을 계산하고, 관련 Plot을 저장.

    Args:
        ppg_signal (np.ndarray): 분석할 Raw PPG 신호 배열.
        show (bool, optional): 분석 Plot을 화면에 표시할지 여부.
        title (str, optional): Plot에 표시될 제목.
        save_path (str, optional): Plot을 저장할 경로 및 파일명.

    Returns:
        pd.DataFrame:
            계산된 모든 HRV 지표가 포함된 DataFrame. 오류 발생 시 빈 DataFrame을 반환.
    """
    try:
        # PPG 신호 처리 및 Peak 검출
        processed, info = nk.ppg_process(ppg_signal, sampling_rate=FS_PPG)
        peaks = info.get("PPG_Peaks")

        # 검출된 Peak를 기반으로 HRV 지표 계산 및 시각화
        hrv = nk.hrv(peaks, sampling_rate=FS_PPG, show=show, title=title + ' (HRV)' if title else None, save_path=save_path + ' (HRV)' if save_path else None)

        if show:
            plt.show()

        return hrv

    except Exception as e:
        print(f"[get_PPG_HRV ERROR] {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def calculate_ptt_peak(ecg_file, ppg_file, ecg_col, ppg_col, ptt_range=(150, 450)):
    """
    ECG와 PPG 신호로부터 PTT(Pulse Transit Time)를 계산.

    Args:
        ecg_file (str): ECG 데이터 CSV 파일의 경로.
        ppg_file (str): PPG 데이터 CSV 파일의 경로.
        ecg_col (str): ECG 신호가 저장된 컬럼명.
        ppg_col (str): PPG 신호가 저장된 컬럼명.
        ptt_range (tuple[int, int], optional): 유효한 PTT 값의 범위(ms).

    Returns:
        dict | None:
            PTT 분석 결과(값 리스트, 평균, 표준편차)가 담긴 딕셔너리.
            데이터 로딩 또는 처리 중 오류 발생 시 None을 반환합니다.
    """
    try:
        # 데이터 로드 및 동기화
        # CSV 파일 로드
        df_ecg = pd.read_csv(ecg_file)
        df_ppg = pd.read_csv(ppg_file)

        # UNIX 타임스탬프를 datetime 객체로 변환하고 신호 전처리
        df_ecg['Timestamp'] = pd.to_datetime(df_ecg['Shimmer_820D_Timestamp_Unix_CAL'], unit='ms')
        df_ecg[ecg_col] = nk.ecg_clean(df_ecg[ecg_col], sampling_rate=512)
        df_ppg['Timestamp'] = pd.to_datetime(df_ppg['id95AE_Timestamp_Unix_CAL'], unit='ms')
        df_ppg[ppg_col] = nk.ppg_clean(df_ppg[ppg_col], sampling_rate=51.2)

        # 타임스탬프를 인덱스로 설정하여 두 데이터프레임 병합
        df_ecg = df_ecg.set_index('Timestamp')
        df_ppg = df_ppg.set_index('Timestamp')
        df = pd.concat([df_ecg[ecg_col], df_ppg[ppg_col]], axis=1)

         # 샘플링 레이트가 다른 ppg 신호를 보간(interpolate)하여 맞춤
        df[ppg_col] = df[ppg_col].interpolate(method='cubic')

        # 동기화 후 결측치가 있는 행 제거
        df = df.dropna(subset=[ecg_col, ppg_col])

    except Exception as e:
        print(f"데이터 로딩 오류: {e}")
        return None

    # 동기화된 데이터의 타임스탬프 간격 중앙값을 이용해 실제 샘플링 레이트 계산
    time_diff = df.index.to_series().diff().median().total_seconds()
    sampling_rate = 1 / time_diff

    # --- ECG R-peak 탐지 ---
    signals, info = nk.ecg_process(df[ecg_col], sampling_rate=sampling_rate)
    ecg_peaks = info['ECG_R_Peaks']
    ecg_peak_times = df.index[ecg_peaks] # R-peak 발생 시간

    # --- PPG 처리 ---
    signals, info = nk.ppg_process(df[ppg_col], sampling_rate=sampling_rate)
    ppg_peaks = info["PPG_Peaks"]
    ppg_peak_times = df.index[ppg_peaks] # PPG peak 발생 시간

    # --- PTT 계산 (R-peak → PPG peak) ---
    ptt_peak = []
    for ecg_time in ecg_peak_times:
        # 현재 R-peak 이후에 발생한 PPG peak들만 필터링
        future_peaks = ppg_peak_times[ppg_peak_times > ecg_time]
        if not future_peaks.empty:
            # 가장 먼저 나타나는 PPG peak와의 시간 차이(ms) 계산
            dt = (future_peaks[0] - ecg_time).total_seconds() * 1000
            # 계산된 PTT가 유효한 범위 내에 있는지 확인 후 추가
            if ptt_range[0] < dt < ptt_range[1]:
                ptt_peak.append(dt)

    # 결과 정리
    results = {
        "ptt_peak_values": ptt_peak,
        "ptt_peak_mean": np.mean(ptt_peak) if len(ptt_peak) > 0 else None,
        "ptt_peak_std": np.std(ptt_peak) if len(ptt_peak) > 0 else None,
    }

    return results

def process_subject_PTT(base_dir, subject, save_path = None):
    """
    한 Subject의 모든 세션/파일에 대해 PTT를 계산하고 결과를 DataFrame으로 통합.

    Args:
        base_dir (str): 'ECG'와 'PPG' 폴더를 포함하는 Subject 데이터의 경로.
        subject (str): 현재 처리 중인 Subject의 ID (결과 저장용).
        save_dir (str): PTT를 저장할 디렉토리

    Returns:
        pd.DataFrame:
            Subject의 모든 파일과 ECG 채널별 PTT 분석 결과가 포함된 데이터프레임.
    """
    print(base_dir+' PTT 분석 시작')
    final_data = []

    # 분석할 ECG/PPG 채널명 정의
    ecg_channels = [
        'Shimmer_820D_ECG_LA-RA_24BIT_CAL',
        'Shimmer_820D_ECG_LL-LA_24BIT_CAL',
        'Shimmer_820D_ECG_LL-RA_24BIT_CAL',
        'Shimmer_820D_ECG_Vx-RL_24BIT_CAL'
    ]
    ppg_channel = 'id95AE_PPG_A13_CAL'

    # ECG 및 PPG 데이터의 기본 경로 설정
    ECG_path = os.path.join(base_dir, 'ECG')
    PPG_path = os.path.join(base_dir, 'PPG')

    # 각 디렉토리 파일 순회 (low1.csv, low2.csv, ...)
    for condition_dir in ['start', 'low', 'mid', 'high', 'end']:
        ECG_sub_path = os.path.join(ECG_path, condition_dir)
        PPG_sub_path = os.path.join(PPG_path, condition_dir)

        # 각 조건(start, low, mid, high, end) 폴더 내의 파일 순회
        for file_name in os.listdir(ECG_sub_path):
            ecg_file = os.path.join(ECG_sub_path, file_name)
            ppg_file = os.path.join(PPG_sub_path, file_name)

            # 대응하는 ECG와 PPG 파일이 모두 존재하는지 확인
            if os.path.exists(ecg_file) and os.path.exists(ppg_file):
                # 한 파일의 결과를 저장할 딕셔너리 초기화
                row_data = {
                    'Subject': subject,
                    # 'Condition': condition_dir,
                    'File': file_name
                }
                # 정의된 모든 ECG 채널에 대해 PTT 계산 반복
                for ecg_channel in ecg_channels:
                    try:
                        # PTT 계산 함수 호출
                        results = calculate_ptt_peak(ecg_file, ppg_file, ecg_channel, ppg_channel)
                    except Exception as e:
                        print(f"오류 발생 (파일: {file_name}, 채널: {ecg_channel}): {e}")
                        # 오류 시 결과값 None으로 채우기
                        row_data[f"{ecg_channel.split('_')[-3]}_PTT_avg"] = None
                        row_data[f"{ecg_channel.split('_')[-3]}_PTT_std"] = None
                        continue

                    try:
                        # PTT 평균
                        row_data[f"{ecg_channel.split('_')[-3]}_PTT_avg"] = results['ptt_peak_mean']
                        # PTT 표준편차
                        row_data[f"{ecg_channel.split('_')[-3]}_PTT_std"] = results['ptt_peak_std']
                    except Exception as e :
                        print(file_name, ecg_channel, e)
                        row_data[f"{ecg_channel.split('_')[-3]}_PTT_avg"] = None
                        row_data[f"{ecg_channel.split('_')[-3]}_PTT_std"] = None
                        continue
            # 한 파일에 대한 모든 채널의 분석 결과를 최종 리스트에 추가
            final_data.append(row_data)
    # 모든 파일에 대한 분석 결과 dataframe 변환
    result = pd.DataFrame(final_data)

    # save_path가 있으면 dataframe 저장
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        print(os.path.join(save_path, subject+".csv")+'에 PTT 계산 결과를 저장했습니다.')
        result.to_csv(os.path.join(save_path, subject+".csv"))
    return result

def process_biosignals_rr_si(subject_path, signal_type, save_path=None):
    """
    ECG, PPG 데이터를 처리하여 호흡률(RR)과 스트레스 지수(SI) 추출.

    Args:
        subject_path (str): 분석할 Subject의 데이터가 담긴 경로.
        signal_type (str): 처리할 신호 종류. 'PPG' 또는 'ECG'.
        save_path (str, optional): 결과를 저장할 최상위 디렉토리 경로.

    Returns:
        float | None: 계산된 모든 RR 값들의 전체 평균. 처리된 값이 없으면 None.
    """
    print(subject_path+' '+signal_type+' Signal - SI, RR 분석 시작')
    # 1. 경로 및 파일 목록 설정
    if os.path.isfile(subject_path) and subject_path.endswith(".csv"):
        dir_path = os.path.dirname(subject_path)
        file_list = [os.path.basename(subject_path)]
        is_file_mode = True
    else:
        dir_path = subject_path
        file_list = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
        is_file_mode = False

    path_parts = os.path.normpath(subject_path).split(os.sep)
    subject_name = path_parts[-3] if is_file_mode else path_parts[-2]

    # RR과 SI 결과를 각각 저장할 리스트 초기화
    results_list_rr = []
    results_list_si = []
    all_rr_values = []
    all_si_values = []

    # 모든 하위 폴더 및 파일을 순회
    for subdir_name in file_list:
        subdir_path = os.path.join(dir_path, subdir_name)
        if not os.path.isdir(subdir_path):
            continue

        for file_name in os.listdir(subdir_path):
            file_path = os.path.join(subdir_path, file_name)
            if not file_name.endswith('.csv'):
                continue

            try:
                # CSV 파일 로드 및 신호 추출
                raw_df = pd.read_csv(file_path)
                raw_signals = extract_signal(raw_df, signal_type)

                # wide-format 데이터프레임 구조로 통일
                row_data_rr = {"File": file_name}
                row_data_si = {"File": file_name}

                # 채널별로 순회하며 RR 및 SI 계산
                for channel_name, raw in raw_signals.items():
                    # 신호 종류에 맞는 fs 설정
                    fs = FS_PPG if signal_type == 'PPG' else FS_ECG

                    # RR 및 SI 계산 래퍼 함수 호출
                    RR = calculate_rr_feature(raw, fs=fs, sig_type=signal_type)
                    SI = calculate_si_feature(raw, fs=fs, sig_type=signal_type)

                    # RR 및 SI 결과 저장 (e.g., 'LA-RA_rr', 'LA-RA_si')
                    row_data_rr[f"{channel_name}_rr"] = RR
                    row_data_si[f"{channel_name}_si"] = SI
                    if RR is not None:
                        all_rr_values.append(RR)
                    if SI is not None:
                        all_si_values.append(SI)

                # 한 파일의 결과를 최종 리스트에 저장
                results_list_rr.append(row_data_rr)
                results_list_si.append(row_data_si)

            except Exception as e:
                print(f"Failed to process {file_path}: {e}")
    # 모든 결과를 각각의 데이터프레임으로 변환
    df_rr = pd.DataFrame(results_list_rr)
    df_si = pd.DataFrame(results_list_si)

    # 최종 결과를 CSV 파일로 저장
    if save_path:
        save_dir_rr = os.path.join(save_path, 'RR', signal_type, subject_name)
        os.makedirs(save_dir_rr, exist_ok=True) # 폴더를 먼저 생성
        save_file_rr = os.path.join(save_dir_rr, f'{subject_name}.csv') # 생성된 폴더 안에 파일 경로 지정
        df_rr.to_csv(save_file_rr, index=False)
        print(f"{save_file_rr}에 {signal_type} Signal - RR 분석 결과를 저장했습니다.")

        save_dir_si = os.path.join(save_path, 'SI', signal_type, subject_name)
        os.makedirs(save_dir_si, exist_ok=True) # 폴더를 먼저 생성
        save_file_si = os.path.join(save_dir_si, f'{subject_name}.csv') # 생성된 폴더 안에 파일 경로 지정
        df_si.to_csv(save_file_si, index=False)
        print(f"{save_file_rr}에 {signal_type} Signal - SI 분석 결과를 저장했습니다.\n")

    return np.mean(all_rr_values) if all_rr_values else None

def calculate_si(sig, fs, sig_type):
    """
    ECG 또는 PPG 신호로부터 Baevsky의 스트레스 지수(SI) 계산.

    Args:
        sig (np.ndarray): Raw ECG 또는 PPG 신호 배열.
        fs (float): 신호의 샘플링 레이트 (Hz).
        sig_type (str): 신호 종류 ('ECG' 또는 'PPG').

    Returns:
        float: 계산된 스트레스 지수. 계산 불가 시 NaN.
    """
    try:
        # 신호 종류에 따라 Peak 검출
        if sig_type == "ECG":
            _, m = nk.ecg_process(sig, sampling_rate=fs)
            peak_locs = np.asarray(m['ECG_R_Peaks'], dtype=int)
        elif sig_type == "PPG":
            _, m = nk.ppg_process(sig, sampling_rate=fs)
            peak_locs = np.asarray(m['PPG_Peaks'], dtype=int)

        # IBI (Inter-Beat Interval) 계산
        ibi = np.diff(peak_locs) / fs
        if len(ibi) == 0:
            return float('nan')

        # 스트레스 지수(SI) 계산
        ibi_counter = Counter(ibi) # IBI 갯수 계산
        M0, M0_count = ibi_counter.most_common(1)[0] # IBI의 최빈값(Mode)
        AM0 = (M0_count / len(ibi)) * 100 # 최빈값의 진폭(%)
        MxDMn = max(ibi) - min(ibi) # 변동 범위(Variation Range)
        Stress_Index = sqrt(AM0 / (2 * M0 * MxDMn)) # SI 공식

        return Stress_Index
    except Exception as e:
        print(f"[heartpy error] {e}")
        return float('nan')

def calculate_rr(sig, fs, sig_type):
    """
    ECG 또는 PPG 신호의 IBI로부터 호흡률(RR) 계산.

    Args:
        sig (np.ndarray): Raw ECG 또는 PPG 신호 배열.
        fs (float): 신호의 샘플링 레이트 (Hz).
        sig_type (str): 신호 종류 ('ECG' 또는 'PPG').

    Returns:
        float | None: 분당 호흡수(breaths/min). 계산 불가 시 None.
    """
    # 최소 신호 길이 확인
    if len(sig) < fs * 5:
        print("Warning: Input signal is too short for RR calculation (less than 5 seconds).")
        return None

    # 신호 표준화 및 스무딩
    if np.std(sig) > 0:
        standardized = (sig - np.mean(sig)) / np.std(sig)
    else:
        standardized = sig # 신호가 평평할 경우 전처리 하지 않음
    smoothed = np.convolve(standardized, np.ones(5) / 5, mode='same')

    # peak 검출
    try:
        if sig_type == "ECG":
            _, m = nk.ecg_process(smoothed, sampling_rate=fs)
            peaks = m['ECG_R_Peaks']
        elif sig_type == "PPG":
            _, m = nk.ppg_process(smoothed, sampling_rate=fs)
            peaks = m['PPG_Peaks']

        # 피크 개수 확인 4개 미만이면 None
        if len(peaks) < 4:
            print("Warning: Not enough peaks found to calculate reliable RR.")
            return None

        peak_times = np.asarray(peaks) / fs

    except Exception as e:
        print(f"Peak detection failed: {e}")
        return None

    # IBI 계산 및 필터링
    ibi = np.diff(peak_times)
    ibi_times = peak_times[1:]

    # 생리학적으로 유효한 IBI 범위
    valid_mask = (ibi >= 0.4) & (ibi <= 1.33)
    ibi = ibi[valid_mask]
    ibi_times = ibi_times[valid_mask]

    # 필터링 후 IBI 개수 4개 미만이면 None
    if len(ibi) < 4:
        print("Warning: Not enough valid IBIs after filtering.")
        return None

    # Lomb-Scargle Periodogram을 이용한 호흡 주파수 추정
    try:
        # 분석할 주파수 범위
        freqs = np.linspace(0.05, 1.5, 2000)
        angular_freqs = 2 * np.pi * freqs
        ibi_mean_removed = ibi - np.mean(ibi)
        psd = lombscargle(ibi_times, ibi_mean_removed, angular_freqs)

        # 호흡 대역(HF: 0.15-0.4Hz)에서 가장 강한 주파수 탐색
        hf_mask = (freqs >= 0.15) & (freqs <= 0.4)
        hf_freqs = freqs[hf_mask]
        hf_psd = psd[hf_mask]
        if len(hf_psd) == 0:
            return None

        peak_idx = np.argmax(hf_psd)
        peak_freq = hf_freqs[peak_idx]

        # 분당 호흡수로 변환
        rr_bpm = peak_freq * 60

    except Exception as e:
        print(f"Lomb-Scargle calculation failed: {e}")
        rr_bpm = None

    return rr_bpm

def calculate_rr_feature(signal, fs, sig_type):
    """
    calculate_rr 함수의 예외처리를 당담
    """
    try:
        rr = calculate_rr(signal, fs, sig_type)
        return rr
    except Exception as e:
        print(f"[ERROR - get_{sig_type}_RR] {e}")
        return None

def calculate_si_feature(signal, fs, sig_type):
    """
    calculate_si 함수의 예외처리를 당담
    """
    try:
        si = calculate_si(signal, fs, sig_type)
        return si
    except Exception as e:
        print(f"[ERROR - get_{sig_type}_SI] {e}")
        return None

def process_biosignals_gsr(subject_path, save_path=None):
    """
    한 Subject의 모든 GSR(EDA) 파일을 분석하여 EDA 지표를 계산하고 결과 반환.

    Args:
        subject_path (str): 분석할 Subject의 데이터가 담긴 경로.
        save_path (str, optional): 분석 결과를 저장할 디렉토리 경로.

    Returns:
        pd.DataFrame | None:
            Subject의 모든 파일에 대한 EDA 분석 결과가 포함된 데이터프레임.
            처리할 데이터가 없는 경우 None을 반환.
    """
    print(subject_path+' GSR Signal - 분석 시작')
    # 입력 경로가 파일인지 폴더인지 확인하고, 처리할 파일 목록 설정
    if os.path.isfile(subject_path) and subject_path.endswith(".csv"):
        file_list = [os.path.basename(subject_path)]
        is_file_mode = True
    else:
        dir_path = subject_path
        file_list = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
        is_file_mode = False

    # Subject 이름 추출
    path_parts = os.path.normpath(subject_path).split(os.sep)
    subject_name = path_parts[-3] if is_file_mode else path_parts[-2]


    # 최종 결과를 저장할 빈 리스트 초기화
    results_list = []

    # 모든 파일을 순회하는 반복문
    for subdir_name in file_list:
        subdir_path = os.path.join(subject_path, subdir_name)
        for file_name in os.listdir(subdir_path):
            # .csv로 끝나지 않으면 데이터로 인식하지 않고 넘김
            if not file_name.endswith('.csv'):
                continue

            # .csv로 끝났다면 경로 저장
            file_path = os.path.join(subdir_path, file_name)

            # CSV 파일 로드 및 GSR 신호 추출
            raw_df = pd.read_csv(file_path)
            raw_signal_dict = extract_signal(raw_df, data_type = 'GSR')

            # 각 채널별로 EDA 지표 계산
            for channel_name, signal_array in raw_signal_dict.items():
                # SCR, SCL 등 EDA 특징 계산
                eda_metrics_df = calculate_scr_scl(signal_array, fs=51.2, sig_type='GSR')

                # 결과 정리 및 리스트에 추가
                if eda_metrics_df is not None:
                    # DataFrame가 None이 아니라면 결과를 딕셔너리로 변환
                    try:
                        eda_metrics_dict = eda_metrics_df.to_dict(orient='records')[0]
                    except:
                        eda_metrics_dict = eda_metrics_df
                    result_data = {'File': file_name, **eda_metrics_dict}
                    results_list.append(result_data)

    if not results_list:
        print("No data was processed.")
        return None
    # 모든 결과를 하나의 데이터프레임으로 변환
    final_df = pd.DataFrame(results_list)

    # 최종 결과를 CSV 파일로 저장 (save_path가 지정된 경우)
    if save_path:
        save_dir = os.path.join(save_path, 'EDA', subject_name)
        os.makedirs(save_dir, exist_ok=True)
        save_file = os.path.join(save_dir, f'{subject_name}.csv')
        final_df.to_csv(save_file, index=False)
        print(f"{save_file}에 GSR Signal - 분석 결과를 저장했습니다.\n")

    return final_df

def calculate_scr_scl(sig, fs, sig_type):
    """
    GSR 신호로부터 SCR, SCL 및 관련 지표 계산.

    Args:
        sig (np.ndarray): Raw GSR 신호 배열.
        fs (float): 신호의 샘플링 레이트 (Hz).
        sig_type (str): 신호 종류 (현재 'GSR'만 지원).

    Returns:
        pd.DataFrame | None: EDA 분석 결과가 담긴 DataFrame. 오류 시 None.
    """
    # Z-score 정규화
    smoothed = (sig - np.mean(sig)) / np.std(sig)

    try:
        if sig_type == "GSR":
            # GSR 전처리
            gsr_sig, m = nk.eda_process(smoothed, sampling_rate=fs)
            analyze_df = _eda_intervalrelated(gsr_sig, sampling_rate=fs)
            # analyze_df = nk.eda_analyze(gsr_sig, sampling_rate=int(fs))

            return analyze_df

    except Exception as e:
        print(f"Wrong signals: {e}")
        return None

def load_feature_csv(path, subject, drop_cols=None, pivot=False, index_col="File"):
    """
    Feature CSV 파일을 로드하고 기본적인 전처리를 수행.

    Args:
        path (str): 로드할 CSV 파일의 전체 경로.
        drop_cols (list[str], optional): 제거할 컬럼 이름의 리스트.
        pivot (bool, optional): 채널(channel) 기준으로 피벗을 수행할지 여부.
        index_col (str, optional): DataFrame의 인덱스로 설정할 컬럼명.

    Returns:
        pd.DataFrame: 전처리된 Feature 데이터프레임.
    """
    df = pd.read_csv(path)

    # end/start 파일 제거
    if "File" in df.columns:
        df = df[~df['File'].str.contains('end|start')].copy()

    # 불필요한 column 제거
    if drop_cols:
        df = df.drop(columns=drop_cols, errors='ignore')

    # pivot 수행
    if pivot and "channel" in df.columns:
        df = df.pivot(
            index="File",
            columns="channel",
            values=[c for c in df.columns if c not in ["File", "channel"]]
        )
        # 다중 컬럼 평탄화
        df.columns = [f"{channel}_{feature}" for feature, channel in df.columns]
        df = df.reset_index()

    # index 지정
    if index_col in df.columns:
        df = df.set_index(index_col)

    return df

def save_joined_features(subject = "1_1_001_V2", save_dir = "../features/joined", feature_path="../features/"):
    """
    특정 Subject의 모든 분산된 Feature들을 하나로 통합하여 단일 CSV 파일로 저장.

    Args:
        subject (str, optional): 처리할 Subject의 ID.
        save_dir (str, optional): 통합된 CSV 파일을 저장할 디렉토리 경로.

    Returns:
        None:
            이 함수는 값을 반환하지 않고, 결과를 파일로 저장.
    """
    print(f"{subject} 추출한 특징 파일 병합 시작")
    # ---- 메인 코드 ----
    EDA_path    = feature_path+"EDA"
    HR_HRV_path = feature_path+"HR_HRV"
    PTT_path    = feature_path+"PTT"
    RR_path     = feature_path+"RR"
    SI_path     = feature_path+"SI"
    label_path  = feature_path+"label"

    # 1. EDA
    EDA_feature = load_feature_csv(f"{EDA_path}/{subject}/{subject}.csv", subject)
    # print(EDA_feature.shape)

    # 2. HR_HRV ECG
    HR_HRV_ECG_feature = load_feature_csv(f"{HR_HRV_path}/ECG/{subject}/{subject}.csv", subject, pivot=True)
    # print(HR_HRV_ECG_feature.shape)

    # 3. HR_HRV PPG
    HR_HRV_PPG_feature = load_feature_csv(f"{HR_HRV_path}/PPG/{subject}/{subject}.csv", subject, pivot=True)
    # print(HR_HRV_PPG_feature.shape)

    # 4. PTT
    PTT_feature = load_feature_csv(f"{PTT_path}/{subject}.csv", subject, drop_cols=[ "Subject"])
    # print(PTT_feature.shape)

    # 5. RR ECG
    RR_ECG_feature = load_feature_csv(f"{RR_path}/ECG/{subject}/{subject}.csv", subject)
    # print(RR_ECG_feature.shape)

    # 6. RR PPG
    RR_PPG_feature = load_feature_csv(f"{RR_path}/PPG/{subject}/{subject}.csv", subject)
    # print(RR_PPG_feature.shape)

    # 7. SI ECG
    SI_ECG_feature = load_feature_csv(f"{SI_path}/ECG/{subject}/{subject}.csv", subject)
    # print(SI_ECG_feature.shape)

    # 8. SI PPG
    SI_PPG_feature = load_feature_csv(f"{SI_path}/PPG/{subject}/{subject}.csv", subject)
    # print(SI_PPG_feature.shape)

    # ---- 모든 feature 병합 ----
    joined_df = (
        EDA_feature
        .join(HR_HRV_ECG_feature, how="outer")
        .join(HR_HRV_PPG_feature, how="outer")
        .join(PTT_feature, how="outer")
        .join(RR_ECG_feature, how="outer")
        .join(RR_PPG_feature, how="outer")
        .join(SI_ECG_feature, how="outer")
        .join(SI_PPG_feature, how="outer")
    )

    # 정렬 우선순위 정의
    order = {"low": 0, "mid": 1, "high": 2}

    def sort_key(fname):

        name = str(fname).lower()
        # group (low/mid/high)
        group = None
        for k in order:
            if k in name:
                group = k
                break
        # 파일명에서 숫자 추출
        m = re.search(r'(\d+)', name)
        num = int(m.group(1)) if m else 0
        return (order.get(group, 99), num)

    # 행 인덱스 정렬
    joined_df = joined_df.reindex(sorted(joined_df.index, key=sort_key))

    # print(joined_df)

    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"{subject}.csv")

    # CSV 저장
    joined_df.to_csv(save_path, index=True)

    print(f"{save_path}에 병합된 특징 CSV 파일을 저장하였습니다.\n")

def load_subject_data(fname, joined_dir = r"../features/joined", label_dir = r"../features/label"):
    """
    한 subject의 joined 데이터와 label 데이터를 불러와 병합 준비.

    Args:
        fname (str): CSV 파일 이름 (예: "sub1.csv").
        joined_dir (str): joined feature 파일들이 있는 디렉토리 경로.
        label_dir (str): label 파일들이 있는 디렉토리 경로.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame] | None:
            joined_df (DataFrame): feature 데이터.
            label_df (DataFrame): label 데이터 (File, Q1, Q2만 포함).
            파일이 존재하지 않으면 None 반환.
    """
    print(fname+' 특징과 Label을 로드합니다.')
    # joined feature 파일 경로
    jp = os.path.join(joined_dir, fname)
    # label 파일 경로
    lp = os.path.join(label_dir, fname)

    # label 파일이 없으면 None 반환 (joined만 있는 경우 무시)
    if not os.path.exists(lp):
        return None

    # joined feature CSV 로드
    joined_df = pd.read_csv(jp)
    # label CSV 로드 (File, Q1, Q2만 추출)
    label_df  = pd.read_csv(lp)

    # feature DataFrame, label DataFrame 반환
    return joined_df, label_df[["File", "Q1", "Q2"]]


def process_labels(label_df, label_name="label", method="mean", cut_bins=None):
    """
    label 데이터(Q1, Q2 기반)를 처리하여 단일 label 컬럼 생성.

    Args:
        label_df (pd.DataFrame): label 데이터프레임. 반드시 "Q1", "Q2" 컬럼 포함.
        label_name (str): 최종 label 컬럼명 (default: "label").
        method (str): Q1/Q2 집계 방식. {"mean", "min", "max"} 중 선택.
        cut_bins (list[float] | None): label을 구간화할 구간 리스트.
                                       예: [-np.inf, 1, 3, np.inf] → 클래스 0/1/2.

    Returns:
        pd.DataFrame: "File" 과 label 컬럼(label_name)만 포함된 DataFrame.
    """

    # Q1, Q2 기반 라벨 계산 (평균 / 최소 / 최대)
    if method == "mean":
        label_df[label_name] = label_df[["Q1", "Q2"]].mean(axis=1)
    elif method == "min":
        label_df[label_name] = label_df[["Q1", "Q2"]].min(axis=1)
    elif method == "max":
        label_df[label_name] = label_df[["Q1", "Q2"]].max(axis=1)
    else:
        # 지정되지 않은 method일 경우 예외 발생
        raise ValueError(f"Unknown method: {method}")

    # cut_bins가 주어지면 → 연속형 값을 범주형(클래스)으로 변환
    if cut_bins:
        label_df[label_name] = pd.cut(
            label_df[label_name],  # 변환할 값
            bins=cut_bins,  # 구간 경계값
            labels=list(range(len(cut_bins) - 1))  # 0,1,2,... 라벨링
        ).astype(int)
    print('Label 계산 완료')
    # File, label_name만 반환 (다른 열은 버림)
    return label_df[["File", label_name]]

def get_corr(file_list=None, cut_bins=[-np.inf, 1, 3, np.inf], label_name="label",
             joined_dir=r"../features/joined", label_dir=r"../features/label"):
    """
    각 subject의 feature와 label 간 correlation을 계산하고,
    feature별 평균 correlation을 반환.

    Args:
        file_list (list[str] | None): 분석할 subject 리스트. None이면 전체.
        cut_bins (list[float]): label 구간화 기준 (default: [-inf, 1, 3, inf]).
        label_name (str): label 컬럼명 (default: "label").
        joined_dir (str): feature CSV 디렉토리.
        label_dir (str): label CSV 디렉토리.

    Returns:
        pd.DataFrame:
            corr_df: feature × subject correlation 테이블.
                     + "mean_corr" 컬럼 포함.
    """
    all_corr = []  # subject별 correlation 결과 저장용 리스트

    # -------------------- (1) subject 파일 순회 --------------------
    for fname in os.listdir(joined_dir):
        if not fname.endswith(".csv"):  # CSV 파일만 처리
            continue
        subject = fname[:-4]  # 파일명에서 확장자 제거 → subject ID

        # file_list가 지정되어 있으면 해당 subject만 처리
        if file_list and subject not in file_list:
            continue

        # -------------------- (2) 데이터 로드 --------------------
        data = load_subject_data(fname, joined_dir, label_dir)
        if data is None:
            continue
        joined_df, label_df = data

        # -------------------- (3) 라벨 처리 --------------------
        # Q1, Q2 → method("mean") 방식으로 통합 후,
        # cut_bins 기준으로 클래스화
        processed_label_df = process_labels(
            label_df,
            label_name=label_name,
            method="mean",
            cut_bins=cut_bins
        )

        # -------------------- (4) feature–label correlation 계산 --------------------
        corr_vals = compute_correlations(joined_df, processed_label_df, label_name=label_name)
        if corr_vals is None:
            continue

        # Series에 subject 이름 붙여 저장
        corr_vals.name = subject
        all_corr.append(corr_vals)

    # -------------------- (5) 모든 subject correlation 합치기 --------------------
    corr_df = pd.concat(all_corr, axis=1)

    # -------------------- (6) feature별 mean correlation 추가 --------------------
    valid_counts = corr_df.count(axis=1)  # NaN 제외한 subject 수
    mean_corr = corr_df.mean(axis=1, skipna=True)  # feature별 평균 correlation
    mean_corr[valid_counts < 5] = np.nan  # subject 수가 적으면 NaN 처리
    corr_df["mean_corr"] = mean_corr

    return corr_df


def safe_corr(x, y, min_samples=10):
    """
    NaN/Inf 제거 후 샘플 수와 상수열 여부를 검사하여 Pearson correlation을 계산.
    조건 미충족 시 NaN 반환.

    Args:
        x (pd.Series): feature 벡터.
        y (pd.Series): label 벡터.
        min_samples (int): correlation 계산을 위한 최소 샘플 수.

    Returns:
        float: Pearson correlation 값. 조건 미충족 시 NaN.
    """
    # 두 벡터를 DataFrame으로 묶음
    df_xy = pd.DataFrame({"x": x, "y": y})

    # Inf → NaN 변환 후, NaN 값 제거
    df_xy = df_xy.replace([np.inf, -np.inf], np.nan).dropna()

    # 샘플 수 부족 시 NaN 반환
    if len(df_xy) < min_samples:
        return np.nan

    # x 또는 y가 상수열(값이 모두 동일)일 경우 NaN 반환
    if df_xy["x"].nunique() <= 1 or df_xy["y"].nunique() <= 1:
        return np.nan

    # Pearson correlation 계산
    return df_xy["x"].corr(df_xy["y"])  # 기본값 method='pearson'


def compute_correlations(feature_df, label_df, label_name="label", min_samples=10):
    """
    feature와 label(label_name) 간의 상관계수(correlation)를 계산.

    Args:
        feature_df (pd.DataFrame): feature DataFrame (각 row는 샘플, "File" 컬럼 포함).
        label_df (pd.DataFrame): label DataFrame ("File" + label 컬럼 포함).
        label_name (str): label 컬럼명 (default: "label").
        min_samples (int): 상관계수 계산을 위한 최소 샘플 수.

    Returns:
        pd.Series: 각 feature별 correlation 값 (index: feature명).
                   label 컬럼(label_name)은 제외됨.
    """
    # "File" 기준으로 feature와 label 병합
    df = pd.merge(label_df, feature_df, on="File", how="inner")

    # 수치형 데이터만 선택 + Inf → NaN 처리
    numeric_df = df.select_dtypes(include=["number"]).replace([np.inf, -np.inf], np.nan)

    # label_name 컬럼이 없으면 None 반환
    if label_name not in numeric_df.columns:
        return None

    # label 벡터 추출
    y = numeric_df[label_name]
    corr_vals = {}

    # 각 feature별로 safe_corr 실행
    for col in numeric_df.columns:
        if col == label_name:
            continue
        corr_vals[col] = safe_corr(numeric_df[col], y, min_samples=min_samples)

    # feature별 correlation 결과 반환 (Series)
    return pd.Series(corr_vals)

def eda_autocor(eda_cleaned, sampling_rate=1000, lag=4):
    """**EDA Autocorrelation**

    Compute the autocorrelation measure of raw EDA signal i.e., the correlation between the time
    series data and a specified time-lagged version of itself.

    Parameters
    ----------
    eda_cleaned : Union[list, np.array, pd.Series]
        The cleaned EDA signal.
    sampling_rate : int
        The sampling frequency of raw EDA signal (in Hz, i.e., samples/second). Defaults to 1000Hz.
    lag : int
        Time lag in seconds. Defaults to 4 seconds to avoid autoregressive
        correlations approaching 1, as recommended by Halem et al. (2020).

    Returns
    -------
    float
        Autocorrelation index of the eda signal.

    See Also
    --------
    eda_simulate, eda_clean


    Examples
    ---------
    .. ipython:: python

      import neurokit2 as nk

      # Simulate EDA signal
      eda_signal = nk.eda_simulate(duration=5, scr_number=5, drift=0.1)
      eda_cleaned = nk.eda_clean(eda_signal)
      cor = nk.eda_autocor(eda_cleaned)
      cor

    References
    -----------
    * van Halem, S., Van Roekel, E., Kroencke, L., Kuper, N., & Denissen, J. (2020). Moments that
      matter? On the complexity of using triggers based on skin conductance to sample arousing
      events within an experience sampling framework. European Journal of Personality, 34(5),
      794-807.

    """
    # Sanity checks
    if isinstance(eda_cleaned, pd.DataFrame):
        colnames = eda_cleaned.columns.values
        if len([i for i in colnames if "EDA_Clean" in i]) == 0:
            raise ValueError(
                "NeuroKit error: eda_autocor(): Your input does not contain the cleaned EDA signal."
            )
        else:
            eda_cleaned = eda_cleaned["EDA_Clean"]
    if isinstance(eda_cleaned, pd.Series):
        eda_cleaned = eda_cleaned.values

    # Autocorrelation
    lag_samples = int(lag * sampling_rate)

    if lag_samples > len(eda_cleaned):
        raise ValueError(
            "NeuroKit error: eda_autocor(): The time lag "
            "exceeds the duration of the EDA signal. "
            "Consider using a longer duration of the EDA signal."
        )

    cor, _ = signal_autocor(eda_cleaned, lag=lag_samples)

    return cor

def _eda_intervalrelated(data, output={}, sampling_rate=1000, method_sympathetic="posada", **kwargs):
    """Format input for dictionary."""
    # Sanitize input
    colnames = data.columns.values

    # SCR Peaks
    if "SCR_Peaks" not in colnames:
        warn(
            "We couldn't find an `SCR_Peaks` column. Returning NaN for N peaks.",
            category=NeuroKitWarning,
        )
        output["SCR_Peaks_N"] = np.nan
    else:
        output["SCR_Peaks_N"] = np.nansum(data["SCR_Peaks"].values)

    # Peak amplitude
    if "SCR_Amplitude" not in colnames:
        warn(
            "We couldn't find an `SCR_Amplitude` column. Returning NaN for peak amplitude.",
            category=NeuroKitWarning,
        )
        output["SCR_Peaks_Amplitude_Mean"] = np.nan
    else:
        peaks_idx = data["SCR_Peaks"] == 1
        # Mean amplitude is only computed over peaks. If no peaks, return NaN
        if peaks_idx.sum() > 0:
            output["SCR_Peaks_Amplitude_Mean"] = np.nanmean(data[peaks_idx]["SCR_Amplitude"].values)
        else:
            output["SCR_Peaks_Amplitude_Mean"] = np.nan

    # Get variability of tonic
    if "EDA_Tonic" in colnames:
        output["EDA_Tonic_SD"] = np.nanstd(data["EDA_Tonic"].values)

    # EDA Sympathetic
    output.update({"EDA_Sympathetic": np.nan, "EDA_SympatheticN": np.nan})  # Default values
    if len(data) > sampling_rate * 64:
        if "EDA_Clean" in colnames:
            output.update(
                eda_sympathetic(
                    data["EDA_Clean"],
                    sampling_rate=sampling_rate,
                    method=method_sympathetic,
                )
            )
        elif "EDA_Raw" in colnames:
            # If not clean signal, use raw
            output.update(
                eda_sympathetic(
                    data["EDA_Raw"],
                    sampling_rate=sampling_rate,
                    method=method_sympathetic,
                )
            )

    # EDA autocorrelation
    output.update({"EDA_Autocorrelation": np.nan})  # Default values
    # try:
    #     if len(data) > sampling_rate:  # 30 seconds minimum (NOTE: somewhat arbitrary)
    #         if "EDA_Clean" in colnames:
    #             output["EDA_Autocorrelation"] = eda_autocor(data["EDA_Clean"], sampling_rate=sampling_rate,lag=5)
    #         elif "EDA_Raw" in colnames:
    #             # If not clean signal, use raw
    #             output["EDA_Autocorrelation"] = eda_autocor(data["EDA_Raw"], sampling_rate=sampling_rate,lag=5)
    # except:
    #     pass
    return output