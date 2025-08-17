#!/usr/bin/env python3
"""
Pipeline emulator tests - script style to work with run_tests.py
"""
import sys
import os

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from emotune.core.emotion.pipeline_emulator import PipelineEmulator


def run():
    try:
        # Voice-only basic
        pipe = PipelineEmulator()
        pipe.set_analysis_mode('fusion')
        pipe.set_fusion_options(allow_fallback=True, fusion_min_conf=0.1)
        out = pipe.step(voice={'valence': 0.35, 'arousal': 0.40, 'confidence': 0.38}, face=None)
        fused = out['fused']
        assert fused['confidence'] > 0.0
        assert fused['sources']['voice'] is True
        assert fused['sources']['face'] is False
        print('✓ voice_only_pipeline_basic')

        # Face-only low conf fallback to voice
        pipe = PipelineEmulator()
        pipe.set_analysis_mode('face')
        pipe.set_fusion_options(allow_fallback=True, fusion_min_conf=0.1)
        pipe.set_confidence_thresholds(face=0.6, voice=0.3)
        out = pipe.step(face={'valence': 0.1, 'arousal': 0.1, 'confidence': 0.2}, voice={'valence': -0.2, 'arousal': 0.3, 'confidence': 0.4})
        assert out['fused']['sources']['voice'] is True
        print('✓ face_only_low_conf_fallback_to_voice')

        # Fusion disagreement increases uncertainty
        pipe = PipelineEmulator()
        pipe.set_analysis_mode('fusion')
        pipe.set_fusion_options(allow_fallback=True, fusion_min_conf=0.1)
        out_agree = pipe.step(face={'valence': 0.5, 'arousal': 0.5, 'confidence': 0.8}, voice={'valence': 0.5, 'arousal': 0.5, 'confidence': 0.8})
        out_disagree = pipe.step(face={'valence': 0.8, 'arousal': -0.8, 'confidence': 0.8}, voice={'valence': -0.8, 'arousal': 0.8, 'confidence': 0.8})
        u_agree = out_agree['fused'].get('uncertainty', 0.0)
        u_disagree = out_disagree['fused'].get('uncertainty', 0.0)
        assert u_disagree >= u_agree
        print('✓ fusion_disagreement_increases_uncertainty')

        # No inputs default and predict-only filtered structure
        pipe = PipelineEmulator()
        pipe.set_analysis_mode('fusion')
        out = pipe.step(face=None, voice=None)
        fused = out['fused']
        assert fused['mean']['valence'] == 0.0 and fused['mean']['arousal'] == 0.0
        assert fused['confidence'] == 0.0
        assert 'mean' in out['filtered'] and 'covariance' in out['filtered']
        print('✓ no_inputs_returns_default_then_predict_only')

        return True
    except Exception as e:
        print(f"✗ pipeline_emulator tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    ok = run()
    sys.exit(0 if ok else 1) 