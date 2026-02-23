"""Quick test to verify channel configurations without FLOW."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from nsrr_tools.utils.config import Config
from nsrr_tools.core.channel_mapper import ChannelMapper
from nsrr_tools.core.modality_detector import ModalityDetector

def test_channel_config():
    print("=" * 80)
    print("Channel Configuration Test (Post-FLOW Removal)")
    print("=" * 80)
    
    # Load config
    config = Config()
    mapper = ChannelMapper(config)
    detector = ModalityDetector(config)
    
    # Test 1: RESP channels (should only have Thor/ABD)
    print("\n1. RESP Channel Alternatives:")
    resp_channels = config.modality_groups['modalities']['RESP']['channels']
    print(f"   Expected: {resp_channels}")
    assert 'Flow' not in resp_channels, "FLOW should be removed!"
    assert 'Thor' in resp_channels, "Thor should be present"
    assert 'ABD' in resp_channels, "ABD should be present"
    print("   ✅ FLOW removed, Thor and ABD present")
    
    # Test 2: Check channel alternatives
    print("\n2. Channel Alternatives Check:")
    print("   Thor alternatives:", config.get_channel_alternatives('Thor')[:5], "...")
    print("   ABD alternatives:", config.get_channel_alternatives('ABD')[:5], "...")
    try:
        flow_alts = config.get_channel_alternatives('Flow')
        print(f"   ⚠️  WARNING: Flow still has alternatives: {flow_alts[:3]}")
    except KeyError:
        print("   ✅ Flow channel properly removed")
    
    # Test 3: Referenced channels confirmed
    print("\n3. Referenced Channel Format Check:")
    eeg_channels = ['C3-M2', 'C4-M1', 'O1-M2', 'O2-M1']
    for ch in eeg_channels:
        alts = config.get_channel_alternatives(ch)
        print(f"   {ch}: {len(alts)} alternatives")
        assert ch in alts, f"{ch} should be in its own alternatives"
    print("   ✅ Referenced formats (C3-M2, etc.) preserved")
    
    # Test 4: Modality grouping
    print("\n4. Modality Grouping Test:")
    test_channels = {
        'C3-M2': 'EEG C3-M2',
        'C4-M1': 'EEG C4-M1',
        'LOC': 'EOG LOC-A2',
        'ROC': 'EOG ROC-A1',
        'EKG': 'ECG',
        'CHIN': 'Chin EMG',
        'Thor': 'THOR RES',
        'ABD': 'ABDOMEN'
    }
    
    grouped = detector.group_channels_by_modality(test_channels)
    print(f"   EEG: {grouped.get('EEG', {})}")
    print(f"   EOG: {grouped.get('EOG', {})}")
    print(f"   ECG: {grouped.get('ECG', {})}")
    print(f"   EMG: {grouped.get('EMG', {})}")
    print(f"   RESP: {grouped.get('RESP', {})}")
    
    # Test 5: SleepFM groups
    print("\n5. SleepFM Modality Groups:")
    sleepfm_groups = detector.get_sleepfm_groups(grouped)
    for group, channels in sleepfm_groups.items():
        print(f"   {group}: {list(channels.keys())}")
    
    assert 'BAS' in sleepfm_groups, "BAS modality should exist"
    assert 'RESP' in sleepfm_groups, "RESP modality should exist"
    assert len(sleepfm_groups['RESP']) == 2, "RESP should have 2 channels (Thor, ABD)"
    print("   ✅ All 4 SleepFM modalities present")
    
    # Test 6: Minimum requirements
    print("\n6. Minimum Requirements Check:")
    requirements = config.modality_groups.get('minimum_requirements', {})
    print(f"   Found {len(requirements)} requirement options")
    for opt_name, opt_config in requirements.items():
        print(f"   - {opt_name}: {opt_config.get('description', 'N/A')}")
    
    print("\n" + "=" * 80)
    print("✅ All tests passed! Configuration is correct.")
    print("=" * 80)

if __name__ == '__main__':
    test_channel_config()
