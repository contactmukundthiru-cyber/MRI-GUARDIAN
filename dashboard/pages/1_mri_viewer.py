"""
MRI Reconstruction Viewer Page
Interactive visualization of MRI reconstructions with hallucination detection
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="MRI Viewer | MRI-GUARDIAN", page_icon="🧠", layout="wide")

st.header("🧠 Interactive MRI Reconstruction Viewer")
st.markdown("""
Explore how different reconstruction methods handle MRI data.
Observe how hallucinations appear in black-box methods and how Guardian detects them.
""")


@st.cache_data
def generate_phantom_mri(size: int = 256, add_lesion: bool = True,
                         lesion_size: int = 15, lesion_position: tuple = (0.3, 0.4)) -> np.ndarray:
    """Generate a simplified brain phantom MRI."""
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)

    # Brain outline (ellipse)
    brain = np.exp(-((X/0.8)**2 + (Y/0.6)**2) * 3)

    # Ventricles
    ventricle1 = 0.3 * np.exp(-((X-0.1)**2/0.02 + (Y)**2/0.08) * 10)
    ventricle2 = 0.3 * np.exp(-((X+0.1)**2/0.02 + (Y)**2/0.08) * 10)

    # Gray/white matter contrast
    gm = 0.2 * np.exp(-((X)**2 + (Y)**2) * 2) * np.sin(X*10)**2

    phantom = brain - ventricle1 - ventricle2 + gm

    # Add lesion if requested
    if add_lesion:
        lx, ly = lesion_position
        lesion_radius = lesion_size / size
        lesion = 0.4 * np.exp(-((X-lx)**2 + (Y-ly)**2) / (lesion_radius**2))
        phantom = phantom + lesion

    # Normalize
    phantom = (phantom - phantom.min()) / (phantom.max() - phantom.min())

    return phantom


@st.cache_data
def simulate_kspace(image: np.ndarray) -> np.ndarray:
    """Simulate k-space acquisition."""
    kspace = np.fft.fftshift(np.fft.fft2(image))
    return kspace


@st.cache_data
def undersample_kspace(kspace: np.ndarray, acceleration: int, pattern: str = 'random') -> tuple:
    """Undersample k-space."""
    mask = np.zeros_like(kspace, dtype=bool)
    h, w = kspace.shape

    if pattern == 'random':
        # Random undersampling with center fully sampled
        center_fraction = 0.1
        center_h = int(h * center_fraction / 2)
        center_w = int(w * center_fraction / 2)

        # Always keep center
        mask[h//2-center_h:h//2+center_h, w//2-center_w:w//2+center_w] = True

        # Random sample the rest
        n_samples = int(h * w / acceleration) - mask.sum()
        remaining = ~mask
        indices = np.where(remaining.flatten())[0]
        selected = np.random.choice(indices, size=max(0, n_samples), replace=False)
        mask.flat[selected] = True

    elif pattern == 'cartesian':
        # Cartesian undersampling (every Nth line)
        mask[::acceleration, :] = True
        # Keep center
        center = h // 2
        mask[center-5:center+5, :] = True

    return kspace * mask, mask


def zero_fill_reconstruction(kspace: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Simple zero-fill reconstruction."""
    return np.abs(np.fft.ifft2(np.fft.ifftshift(kspace)))


def simulate_blackbox_reconstruction(kspace: np.ndarray, mask: np.ndarray,
                                     hallucination_strength: float = 0.3) -> tuple:
    """Simulate black-box DL reconstruction with potential hallucinations."""
    # Start with zero-fill
    zf = zero_fill_reconstruction(kspace, mask)

    # Simulate "learned" enhancement
    enhanced = zf.copy()

    # Add some sharpening
    from scipy.ndimage import gaussian_filter
    sharp = zf - gaussian_filter(zf, sigma=2)
    enhanced = enhanced + 0.3 * sharp

    # Simulate hallucination (fake lesion in random location)
    np.random.seed(42)
    h, w = enhanced.shape

    hallucination_map = np.zeros_like(enhanced)

    if hallucination_strength > 0:
        # Add fake lesion
        hx, hy = 0.5 + np.random.uniform(-0.2, 0.2), 0.2 + np.random.uniform(-0.1, 0.1)
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        X, Y = np.meshgrid(x, y)

        fake_lesion = hallucination_strength * np.exp(-((X-hx)**2 + (Y-hy)**2) / 0.003)
        hallucination_map = fake_lesion

        enhanced = enhanced + fake_lesion

    # Add some noise reduction artifacts
    enhanced = gaussian_filter(enhanced, sigma=0.5)

    return np.clip(enhanced, 0, 1), hallucination_map


def simulate_guardian_reconstruction(kspace: np.ndarray, mask: np.ndarray,
                                    full_kspace: np.ndarray) -> np.ndarray:
    """Simulate Guardian physics-guided reconstruction."""
    # Start with zero-fill
    recon = zero_fill_reconstruction(kspace, mask)

    # Iterative refinement with data consistency
    for _ in range(5):
        # Image domain enhancement (simulated)
        from scipy.ndimage import gaussian_filter
        enhanced = recon + 0.1 * (gaussian_filter(recon, sigma=1) - gaussian_filter(recon, sigma=2))

        # Data consistency step (replace measured k-space)
        recon_k = np.fft.fftshift(np.fft.fft2(enhanced))
        recon_k[mask] = kspace[mask]  # Hard data consistency
        recon = np.abs(np.fft.ifft2(np.fft.ifftshift(recon_k)))

    return np.clip(recon, 0, 1)


# Sidebar controls
st.sidebar.header("Simulation Controls")

lesion_present = st.sidebar.checkbox("Add Lesion to Ground Truth", value=True)

if lesion_present:
    lesion_x = st.sidebar.slider("Lesion X Position", -0.5, 0.5, 0.3, 0.05)
    lesion_y = st.sidebar.slider("Lesion Y Position", -0.5, 0.5, 0.4, 0.05)
    lesion_size = st.sidebar.slider("Lesion Size (pixels)", 5, 30, 15, 1)
else:
    lesion_x, lesion_y, lesion_size = 0.3, 0.4, 15

acceleration = st.sidebar.slider("Acceleration Factor", 2, 12, 4, 1)
pattern = st.sidebar.selectbox("Sampling Pattern", ['random', 'cartesian'])
hallucination_strength = st.sidebar.slider("Black-box Hallucination Strength", 0.0, 0.5, 0.3, 0.05)

# Generate images
ground_truth = generate_phantom_mri(
    size=256,
    add_lesion=lesion_present,
    lesion_size=lesion_size,
    lesion_position=(lesion_x, lesion_y)
)

full_kspace = simulate_kspace(ground_truth)
undersampled_kspace, mask = undersample_kspace(full_kspace, acceleration, pattern)

zero_fill = zero_fill_reconstruction(undersampled_kspace, mask)
blackbox, hallucination_map = simulate_blackbox_reconstruction(undersampled_kspace, mask, hallucination_strength)
guardian = simulate_guardian_reconstruction(undersampled_kspace, mask, full_kspace)

# Calculate discrepancy map (Guardian auditing Black-box)
discrepancy = np.abs(blackbox - guardian)
discrepancy_normalized = discrepancy / discrepancy.max() if discrepancy.max() > 0 else discrepancy

# Main display
st.subheader("Reconstruction Comparison")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("**Ground Truth**")
    fig1 = go.Figure(go.Heatmap(z=ground_truth, colorscale='gray', showscale=False))
    fig1.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False, scaleanchor='x'))
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.markdown("**Zero-Fill**")
    fig2 = go.Figure(go.Heatmap(z=zero_fill, colorscale='gray', showscale=False))
    fig2.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False, scaleanchor='x'))
    st.plotly_chart(fig2, use_container_width=True)

with col3:
    st.markdown("**Black-box DL** ⚠️")
    fig3 = go.Figure(go.Heatmap(z=blackbox, colorscale='gray', showscale=False))
    fig3.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False, scaleanchor='x'))
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    st.markdown("**Guardian** ✅")
    fig4 = go.Figure(go.Heatmap(z=guardian, colorscale='gray', showscale=False))
    fig4.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False, scaleanchor='x'))
    st.plotly_chart(fig4, use_container_width=True)

# Quality metrics
st.subheader("Quality Metrics")

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))

def ssim_simple(img1, img2):
    """Simplified SSIM calculation."""
    c1, c2 = 0.01**2, 0.03**2
    mu1, mu2 = img1.mean(), img2.mean()
    sigma1, sigma2 = img1.std(), img2.std()
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
    return ((2*mu1*mu2 + c1) * (2*sigma12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1**2 + sigma2**2 + c2))

metrics_col1, metrics_col2, metrics_col3 = st.columns(3)

with metrics_col1:
    st.metric("Zero-Fill PSNR", f"{psnr(ground_truth, zero_fill):.1f} dB")
    st.metric("Zero-Fill SSIM", f"{ssim_simple(ground_truth, zero_fill):.3f}")

with metrics_col2:
    st.metric("Black-box PSNR", f"{psnr(ground_truth, blackbox):.1f} dB")
    st.metric("Black-box SSIM", f"{ssim_simple(ground_truth, blackbox):.3f}")

with metrics_col3:
    st.metric("Guardian PSNR", f"{psnr(ground_truth, guardian):.1f} dB")
    st.metric("Guardian SSIM", f"{ssim_simple(ground_truth, guardian):.3f}")

# Hallucination detection
st.markdown("---")
st.subheader("🔍 Hallucination Detection")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Hallucination Map (Ground Truth)**")
    fig_h = go.Figure(go.Heatmap(z=hallucination_map, colorscale='Reds', showscale=True))
    fig_h.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                       xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False, scaleanchor='x'))
    st.plotly_chart(fig_h, use_container_width=True)
    st.caption("Where hallucinations were injected")

with col2:
    st.markdown("**Auditor Suspicion Map**")
    fig_d = go.Figure(go.Heatmap(z=discrepancy_normalized, colorscale='YlOrRd', showscale=True))
    fig_d.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                       xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False, scaleanchor='x'))
    st.plotly_chart(fig_d, use_container_width=True)
    st.caption("Where Guardian detected discrepancies")

with col3:
    st.markdown("**Detection Analysis**")

    # Calculate correlation between hallucination and detection
    if hallucination_map.max() > 0:
        correlation = np.corrcoef(hallucination_map.flatten(), discrepancy_normalized.flatten())[0, 1]

        # Detection metrics
        threshold = 0.3
        detected = discrepancy_normalized > threshold
        actual = hallucination_map > 0.1

        tp = np.sum(detected & actual)
        fp = np.sum(detected & ~actual)
        fn = np.sum(~detected & actual)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        st.metric("Detection Correlation", f"{correlation:.3f}")
        st.metric("Precision", f"{precision:.2%}")
        st.metric("Recall", f"{recall:.2%}")
    else:
        st.info("No hallucinations to detect (strength = 0)")

# K-space visualization
st.markdown("---")
st.subheader("📊 K-Space Visualization")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Full K-Space (log magnitude)**")
    kspace_mag = np.log1p(np.abs(full_kspace))
    fig_k = go.Figure(go.Heatmap(z=kspace_mag, colorscale='Viridis', showscale=False))
    fig_k.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                       xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False, scaleanchor='x'))
    st.plotly_chart(fig_k, use_container_width=True)

with col2:
    st.markdown(f"**Sampling Mask ({acceleration}x acceleration)**")
    fig_m = go.Figure(go.Heatmap(z=mask.astype(float), colorscale='Blues', showscale=False))
    fig_m.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                       xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False, scaleanchor='x'))
    st.plotly_chart(fig_m, use_container_width=True)

sampling_percentage = mask.sum() / mask.size * 100
st.caption(f"Sampling rate: {sampling_percentage:.1f}% of k-space acquired")
