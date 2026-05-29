# EfficientNet-B4 Deepfake Detector - Analysis & Bug Report

## Executive Summary
Project analyzed: EfficientNet-B4 deepfake detection system operating on DF40 dataset.

## Current Performance Metrics (per documentation)
- **Accuracy:** 86.45% (target: ≥90%)
- **AUC-ROC:** 0.9447 (good discriminative power)
- **Specificity:** 95.7% (excellent at detecting real images)
- **Sensitivity:** 77.2% (weakness - frequently misses fake images)

## Critical Bugs & Issues

### 1. Windows Compatibility (HIGH PRIORITY)
**Location:** `prepare_df40.py`, lines 131-151  
**Issue:** `os.symlink()` fails on Windows without admin privileges  
**Impact:** Dataset preparation crashes on Windows systems  
**Fix:** Add Windows fallback using `shutil.copy2()`

### 2. Device Detection Flaw (HIGH PRIORITY)
**Location:** `train.py`, lines 185-188  
**Issue:** Hardcodes `torch.cuda.get_device_name(0)` without checking CUDA availability  
**Impact:** Potential crash on systems without CUDA  
**Fix:** Add device availability check

### 3. Checkpoint Loading Issues (HIGH PRIORITY)
**Location:** `train.py`, lines 248-266  
**Issue:** Complex state loading may fail across model versions  
**Impact:** Resume functionality broken  
**Fix:** Simplify checkpoint loading logic

### 4. Memory Management (MEDIUM PRIORITY)
**Location:** `train.py`, line 298  
**Issue:** `torch.cuda.empty_cache()` called unconditionally  
**Impact:** Performance degradation during training  
**Fix:** Add conditional cache clearing

## Performance Issues (Accuracy Enhancement)

### 5. Resolution Too Low (CRITICAL)
**Issue:** Training at 320px instead of native 380px  
**Impact:** Deepfake artefacts (pixel-level) destroyed by downsampling  
**Fix:** Set `IMAGE_SIZE = 380` in config.py  
**Expected Gain:** +3-5% accuracy

### 6. Real Data Bias (CRITICAL)
**Issue:** 100% of "real" images from FaceForensics++ (compressed YouTube videos)  
**Impact:** Model learns "YouTube compression = Real" bias  
**Fix:** Inject Celeb-DF or raw photography datasets into `real/` folder  
**Expected Gain:** +2-4% accuracy

### 7. Class Imbalance Sensitivity (CRITICAL)
**Issue:** Model over-indexes on real images (sensitivity 77.2%)  
**Impact:** Frequently misses fake images  
**Fix:** Adjust class weights, increase fake samples  
**Expected Gain:** +3-5% sensitivity

### 8. Insufficient Training Data
**Issue:** MAX_FAKE_PER_METHOD limited to 5000  
**Impact:** Model doesn't see sufficient variations  
**Fix:** Increase to 10000 per method  
**Expected Gain:** +1-3% accuracy

### 9. Inadequate Head Warmup
**Issue:** Only 3 epochs of head-only training  
**Impact:** Head hasn't learned useful gradient direction  
**Fix:** Increase `UNFREEZE_EPOCH` to 5  
**Expected Gain:** +1-2% accuracy

### 10. Limited Augmentation
**Issue:** Augmentation doesn't include advanced techniques  
**Impact:** Model overfits to specific artefact patterns  
**Fix:** Add CutMix, channel swaps, extreme color jitter  
**Expected Gain:** +2-3% accuracy

## Recommended Action Plan

### Phase 1: Quick Wins (1-2 weeks)
1. ✅ Fix Windows symlink support
2. ✅ Restore native 380px resolution  
3. ✅ Extend head warmup to 5 epochs
4. ✅ Add Celeb-DF real images
5. ✅ Tune classification threshold (current: 0.38)

### Phase 2: Medium Term (2-3 weeks)
1. ✅ Increase dataset capacity (10000/method)
2. ✅ Add CutMix augmentation
3. ✅ Implement model ensemble (3 models)
4. ✅ Advanced LR scheduling

### Phase 3: Advanced (3-4 weeks)
1. ✅ SE blocks in classification head
2. ✅ Confidence-based rejection
3. ✅ Multi-scale feature fusion
4. ✅ Test-time augmentation enhancement

## Expected Outcomes

**Accuracy Trajectory:**
- Current: 86.45%
- Phase 1 completion: 90-92%
- Phase 2 completion: 92-94%
- Phase 3 completion: 94-96%

**Sensitivity Improvement:** Target >85% (current: 77.2%)
**Specificivity Maintenance:** >90% (current: 95.7%)

## Technical Architecture Notes

### Why EfficientNet-B4?
- Compound scaling provides optimal accuracy-per-parameter  
- SE blocks provide channel-wise feature selection
- ImageNet pretraining gives strong foundation

### Why 380px Resolution?
- Deepfake artefacts are pixel-level
- Downsampling to 320px destroys crucial forensic evidence
- Native resolution preserves fine-grained manipulation patterns

### Why Two-Layer Head?
- Strong regularization prevents overfitting across 40 methods
- Dropout (0.4, 0.3) provides necessary generalisation
- GELU activation provides smooth non-linearity

## Immediate Actions Required

1. **P0 Priority:** Fix Windows symlink support (blocking basic functionality)
2. **P0 Priority:** Restore native 380px resolution (critical for detection accuracy)
3. **P1 Priority:** Add Celeb-DF real images (removes data bias)
4. **P1 Priority:** Extend head warmup to 5 epochs
5. **P2 Priority:** Implement ensemble methods (immediate accuracy boost)

## Monitoring Recommendations

Track these metrics after each phase:
- Accuracy (primary success metric)
- AUC-ROC (robustness indicator)
- Sensitivity (fake detection rate)
- Specificity (real acceptance rate)
- Training/validation loss gap (overfitting indicator)