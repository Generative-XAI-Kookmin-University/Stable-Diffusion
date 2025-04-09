import torch
import argparse
import os
from collections import defaultdict

def find_ema_keys(dictionary, prefix=''):
    """
    체크포인트 딕셔너리에서 EMA 관련 키를 재귀적으로 찾습니다.
    """
    ema_keys = []
    for k, v in dictionary.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if 'ema' in k.lower():
            ema_keys.append(full_key)
        if isinstance(v, dict):
            ema_keys.extend(find_ema_keys(v, full_key))
    return ema_keys

def analyze_checkpoint(checkpoint_path):
    """
    주어진 체크포인트 파일을 분석하여 EMA 관련 정보를 출력합니다.
    """
    print(f"Analyzing checkpoint: {checkpoint_path}")
    
    try:
        # 체크포인트 로드
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print("\n1. 체크포인트 최상위 키:")
        for key in checkpoint.keys():
            print(f"   - {key}")
        
        # EMA 관련 최상위 키 확인
        ema_top_keys = [k for k in checkpoint.keys() if 'ema' in k.lower()]
        if ema_top_keys:
            print("\n2. EMA 관련 최상위 키:")
            for key in ema_top_keys:
                print(f"   - {key}")
        else:
            print("\n2. 최상위 레벨에서 EMA 관련 키를 찾을 수 없습니다.")
        
        # state_dict 내에서 EMA 관련 키 확인
        if 'state_dict' in checkpoint:
            state_dict_keys = list(checkpoint['state_dict'].keys())
            print(f"\n3. state_dict 내 키 개수: {len(state_dict_keys)}")
            
            # state_dict 내의 EMA 관련 키 찾기
            ema_state_dict_keys = [k for k in state_dict_keys if 'ema' in k.lower()]
            if ema_state_dict_keys:
                print("\n4. state_dict 내 EMA 관련 키:")
                for key in ema_state_dict_keys[:10]:  # 처음 10개만 출력
                    print(f"   - {key}")
                if len(ema_state_dict_keys) > 10:
                    print(f"   ... 총 {len(ema_state_dict_keys)}개 중 10개만 표시")
            else:
                print("\n4. state_dict 내에서 EMA 관련 키를 찾을 수 없습니다.")
        
            # EMA 접두사 및 접미사 패턴 분석
            ema_prefixes = defaultdict(int)
            for key in state_dict_keys:
                if 'ema' in key.lower():
                    parts = key.split('.')
                    for part in parts:
                        if 'ema' in part.lower():
                            ema_prefixes[part] += 1
            
            if ema_prefixes:
                print("\n5. EMA 키 패턴 분석:")
                for pattern, count in ema_prefixes.items():
                    print(f"   - 패턴 '{pattern}': {count}개 발견")
        else:
            print("\n3. 체크포인트에 'state_dict' 키가 없습니다.")
        
        # 이중 가중치 체크 (일반 가중치와 EMA 가중치)
        if 'state_dict' in checkpoint and ema_state_dict_keys:
            print("\n6. 일반 가중치와 EMA 가중치 비교:")
            
            # EMA 키에서 접두사 제거하여 원본 키 추정
            ema_to_original = {}
            for ema_key in ema_state_dict_keys:
                # 일반적인 EMA 네이밍 패턴에 대한 추정
                if ema_key.startswith('ema_'):
                    original_key = ema_key[4:]  # 'ema_' 제거
                    ema_to_original[ema_key] = original_key
                elif '.ema.' in ema_key:
                    original_key = ema_key.replace('.ema.', '.')
                    ema_to_original[ema_key] = original_key
            
            # 원본-EMA 쌍 확인
            found_pairs = 0
            for ema_key, original_key in ema_to_original.items():
                if original_key in state_dict_keys:
                    found_pairs += 1
                    if found_pairs <= 5:  # 처음 5쌍만 출력
                        print(f"   - 쌍 발견: {original_key} ↔ {ema_key}")
            
            if found_pairs > 0:
                print(f"   총 {found_pairs}개의 일반-EMA 가중치 쌍 발견")
                
                # 임의의 쌍에 대해 값 비교
                if found_pairs > 0:
                    sample_original = list(ema_to_original.values())[0]
                    sample_ema = list(ema_to_original.keys())[0]
                    if sample_original in state_dict_keys:
                        original_tensor = checkpoint['state_dict'][sample_original]
                        ema_tensor = checkpoint['state_dict'][sample_ema]
                        print(f"\n7. 샘플 텐서 비교 ({sample_original} vs {sample_ema}):")
                        print(f"   - 원본 텐서 형태: {original_tensor.shape}")
                        print(f"   - EMA 텐서 형태: {ema_tensor.shape}")
                        print(f"   - 값이 동일한가? {torch.all(original_tensor == ema_tensor).item()}")
                        if not torch.all(original_tensor == ema_tensor).item():
                            diff = (original_tensor - ema_tensor).abs()
                            print(f"   - 평균 차이: {diff.mean().item()}")
                            print(f"   - 최대 차이: {diff.max().item()}")
            else:
                print("   일반 가중치와 매칭되는 EMA 가중치 쌍을 찾을 수 없습니다.")
        
        # EMA 파라미터 확인
        ema_params = {}
        for key in checkpoint.keys():
            if isinstance(checkpoint[key], (int, float)) and 'ema' in key.lower():
                ema_params[key] = checkpoint[key]
        
        if ema_params:
            print("\n8. EMA 관련 파라미터:")
            for key, value in ema_params.items():
                print(f"   - {key}: {value}")
        
        # 결론
        has_ema = bool(ema_top_keys or (
            'state_dict' in checkpoint and ema_state_dict_keys
        ))
        
        print("\n결론:")
        if has_ema:
            print("✅ 이 체크포인트에는 EMA 관련 가중치가 포함되어 있습니다.")
        else:
            print("❌ 이 체크포인트에서 EMA 관련 가중치를 찾을 수 없습니다.")
        
        # EMA 사용 방법 제안
        if has_ema:
            print("\nEMA 가중치 사용 방법:")
            if 'ema_state_dict' in checkpoint:
                print("1. 체크포인트에 'ema_state_dict'가 있습니다. 다음과 같이 로드할 수 있습니다:")
                print("   model.load_state_dict(checkpoint['ema_state_dict'])")
            elif ema_state_dict_keys:
                print("1. 체크포인트의 'state_dict'에 EMA 키가 있습니다. 다음과 같이 처리할 수 있습니다:")
                print("   ema_state_dict = {k: v for k, v in checkpoint['state_dict'].items() if 'ema' in k.lower()}")
                print("   # 필요한 경우 키 이름 변환 후 모델에 로드")
        
    except Exception as e:
        print(f"체크포인트 분석 중 오류가 발생했습니다: {e}")

def main():
    parser = argparse.ArgumentParser(description='체크포인트 파일에서 EMA 가중치 검출')
    parser.add_argument('--checkpoint', type=str, required=True, help='체크포인트 파일 경로')
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"오류: 체크포인트 파일이 존재하지 않습니다: {args.checkpoint}")
        return
    
    analyze_checkpoint(args.checkpoint)

if __name__ == '__main__':
    main()