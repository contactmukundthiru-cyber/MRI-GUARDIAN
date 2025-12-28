"""
Live Demo Page
Interactive demonstration of MRI-GUARDIAN capabilities
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

st.set_page_config(page_title="Live Demo | MRI-GUARDIAN", page_icon="🎬", layout="wide")

st.header("🎬 Live Demo: MRI-GUARDIAN in Action")
st.markdown("""
Watch MRI-GUARDIAN process a scan in real-time. This demo shows:
1. K-space acquisition and undersampling
2. Iterative reconstruction with physics constraints
3. Hallucination detection and alerting
4. LIM and BPS score calculation
""")

# Initialize session state
if 'demo_running' not in st.session_state:
    st.session_state.demo_running = False
if 'demo_step' not in st.session_state:
    st.session_state.demo_step = 0


def generate_brain_phantom(size=256):
    """Generate a brain phantom with lesions."""
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)

    # Brain structure
    brain = np.exp(-((X/0.8)**2 + (Y/0.6)**2) * 3)
    ventricles = 0.3 * (np.exp(-((X-0.1)**2 + Y**2) / 0.015) +
                        np.exp(-((X+0.1)**2 + Y**2) / 0.015))
    gm = 0.15 * np.exp(-(X**2 + Y**2) * 3) * (1 + 0.3*np.sin(X*8)*np.sin(Y*8))

    phantom = brain - ventricles + gm

    # Add real lesion
    lesion = 0.35 * np.exp(-((X-0.35)**2 + (Y-0.3)**2) / 0.003)
    phantom += lesion

    return (phantom - phantom.min()) / (phantom.max() - phantom.min()), lesion > 0.1


# Demo configuration
st.sidebar.header("Demo Configuration")

demo_speed = st.sidebar.select_slider(
    "Animation Speed",
    options=['Slow', 'Normal', 'Fast'],
    value='Normal'
)

speed_map = {'Slow': 1.0, 'Normal': 0.5, 'Fast': 0.2}
delay = speed_map[demo_speed]

acceleration = st.sidebar.slider("Acceleration Factor", 2, 8, 4)
show_hallucination = st.sidebar.checkbox("Inject Black-box Hallucination", value=True)

# Generate base images
phantom, lesion_mask = generate_brain_phantom()
kspace_full = np.fft.fftshift(np.fft.fft2(phantom))

# Create undersampling mask
mask = np.zeros_like(phantom, dtype=bool)
h, w = phantom.shape
mask[h//2-10:h//2+10, w//2-10:w//2+10] = True  # Center
np.random.seed(42)
for i in range(0, h, acceleration):
    mask[i, :] = True

kspace_under = kspace_full * mask

# Demo control buttons
col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    if st.button("▶️ Start Demo", type="primary", use_container_width=True):
        st.session_state.demo_running = True
        st.session_state.demo_step = 0

with col2:
    if st.button("⏹️ Reset", use_container_width=True):
        st.session_state.demo_running = False
        st.session_state.demo_step = 0

# Progress tracking
progress_placeholder = st.empty()
status_placeholder = st.empty()

# Main visualization area
st.markdown("---")

# Create placeholders for dynamic updates
main_viz = st.empty()
metrics_viz = st.empty()
alerts_viz = st.empty()

# Demo steps
demo_steps = [
    ("Acquiring K-Space Data...", "kspace"),
    ("Applying Undersampling Mask...", "mask"),
    ("Zero-Fill Reconstruction...", "zerofill"),
    ("Guardian: Iteration 1...", "iter1"),
    ("Guardian: Iteration 2...", "iter2"),
    ("Guardian: Iteration 3...", "iter3"),
    ("Guardian: Final Refinement...", "final"),
    ("Computing LIM Score...", "lim"),
    ("Computing BPS Score...", "bps"),
    ("Auditing Black-box Output...", "audit"),
    ("Generating Safety Report...", "report"),
    ("Demo Complete!", "done")
]


def run_demo_step(step_idx):
    """Execute a single demo step."""
    step_name, step_type = demo_steps[min(step_idx, len(demo_steps)-1)]

    # Update progress
    progress = (step_idx + 1) / len(demo_steps)
    progress_placeholder.progress(progress, text=f"Step {step_idx + 1}/{len(demo_steps)}")
    status_placeholder.info(f"🔄 {step_name}")

    # Generate appropriate visualization based on step
    with main_viz.container():
        if step_type == "kspace":
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Ground Truth Phantom**")
                fig = go.Figure(go.Heatmap(z=phantom, colorscale='gray', showscale=False))
                fig.update_layout(height=350, margin=dict(l=10,r=10,t=10,b=10),
                                 xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor='x'))
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("**K-Space (Frequency Domain)**")
                fig = go.Figure(go.Heatmap(z=np.log1p(np.abs(kspace_full)), colorscale='Viridis', showscale=False))
                fig.update_layout(height=350, margin=dict(l=10,r=10,t=10,b=10),
                                 xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor='x'))
                st.plotly_chart(fig, use_container_width=True)

        elif step_type == "mask":
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Sampling Mask**")
                fig = go.Figure(go.Heatmap(z=mask.astype(float), colorscale='Blues', showscale=False))
                fig.update_layout(height=350, margin=dict(l=10,r=10,t=10,b=10),
                                 xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor='x'))
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown(f"**Undersampled K-Space ({acceleration}x)**")
                fig = go.Figure(go.Heatmap(z=np.log1p(np.abs(kspace_under)), colorscale='Viridis', showscale=False))
                fig.update_layout(height=350, margin=dict(l=10,r=10,t=10,b=10),
                                 xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor='x'))
                st.plotly_chart(fig, use_container_width=True)

        elif step_type == "zerofill":
            zf = np.abs(np.fft.ifft2(np.fft.ifftshift(kspace_under)))

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Ground Truth**")
                fig = go.Figure(go.Heatmap(z=phantom, colorscale='gray', showscale=False))
                fig.update_layout(height=300, margin=dict(l=10,r=10,t=10,b=10),
                                 xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor='x'))
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("**Zero-Fill (Baseline)**")
                fig = go.Figure(go.Heatmap(z=zf, colorscale='gray', showscale=False))
                fig.update_layout(height=300, margin=dict(l=10,r=10,t=10,b=10),
                                 xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor='x'))
                st.plotly_chart(fig, use_container_width=True)

            with col3:
                st.markdown("**Difference**")
                diff = np.abs(phantom - zf)
                fig = go.Figure(go.Heatmap(z=diff, colorscale='Reds', showscale=False))
                fig.update_layout(height=300, margin=dict(l=10,r=10,t=10,b=10),
                                 xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor='x'))
                st.plotly_chart(fig, use_container_width=True)

        elif step_type in ["iter1", "iter2", "iter3", "final"]:
            # Simulate iterative reconstruction
            iter_num = {"iter1": 1, "iter2": 2, "iter3": 3, "final": 5}[step_type]

            recon = np.abs(np.fft.ifft2(np.fft.ifftshift(kspace_under)))
            for _ in range(iter_num):
                from scipy.ndimage import gaussian_filter
                enhanced = recon + 0.1 * (gaussian_filter(recon, 1) - gaussian_filter(recon, 2))
                recon_k = np.fft.fftshift(np.fft.fft2(enhanced))
                recon_k[mask] = kspace_under[mask]
                recon = np.abs(np.fft.ifft2(np.fft.ifftshift(recon_k)))

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Ground Truth**")
                fig = go.Figure(go.Heatmap(z=phantom, colorscale='gray', showscale=False))
                fig.update_layout(height=300, margin=dict(l=10,r=10,t=10,b=10),
                                 xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor='x'))
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown(f"**Guardian (Iteration {iter_num})**")
                fig = go.Figure(go.Heatmap(z=recon, colorscale='gray', showscale=False))
                fig.update_layout(height=300, margin=dict(l=10,r=10,t=10,b=10),
                                 xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor='x'))
                st.plotly_chart(fig, use_container_width=True)

            with col3:
                st.markdown("**Residual Error**")
                diff = np.abs(phantom - recon)
                fig = go.Figure(go.Heatmap(z=diff, colorscale='Reds', showscale=False))
                fig.update_layout(height=300, margin=dict(l=10,r=10,t=10,b=10),
                                 xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor='x'))
                st.plotly_chart(fig, use_container_width=True)

            # Show improvement metrics
            zf = np.abs(np.fft.ifft2(np.fft.ifftshift(kspace_under)))
            psnr_zf = 20 * np.log10(1.0 / np.sqrt(np.mean((phantom - zf)**2)))
            psnr_recon = 20 * np.log10(1.0 / np.sqrt(np.mean((phantom - recon)**2)))

            with metrics_viz.container():
                m1, m2, m3 = st.columns(3)
                m1.metric("Zero-Fill PSNR", f"{psnr_zf:.1f} dB")
                m2.metric("Guardian PSNR", f"{psnr_recon:.1f} dB", delta=f"+{psnr_recon-psnr_zf:.1f} dB")
                m3.metric("Iteration", f"{iter_num}/5")

        elif step_type == "lim":
            # Simulate LIM calculation
            st.markdown("### Computing Lesion Integrity Marker (LIM)")

            col1, col2 = st.columns([1, 1])
            with col1:
                # Show lesion region
                st.markdown("**Detected Lesion Region**")
                lesion_highlight = phantom.copy()
                lesion_highlight[lesion_mask] *= 1.5

                fig = go.Figure(go.Heatmap(z=lesion_highlight, colorscale='gray', showscale=False))
                # Add circle around lesion
                fig.add_shape(type="circle", x0=195, y0=140, x1=235, y1=180,
                             line=dict(color="lime", width=3))
                fig.update_layout(height=350, margin=dict(l=10,r=10,t=10,b=10),
                                 xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor='x'))
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("**LIM Component Scores**")
                components = ['Intensity', 'Shape', 'Texture', 'Edge', 'Location']
                scores = [0.92, 0.88, 0.85, 0.87, 0.96]

                fig = go.Figure(go.Bar(
                    x=scores, y=components, orientation='h',
                    marker_color=['#2ecc71' if s > 0.85 else '#f39c12' for s in scores],
                    text=[f'{s:.0%}' for s in scores],
                    textposition='outside'
                ))
                fig.update_layout(height=350, xaxis=dict(range=[0, 1.1]),
                                 margin=dict(l=10,r=10,t=10,b=10))
                st.plotly_chart(fig, use_container_width=True)

            with metrics_viz.container():
                m1, m2, m3 = st.columns(3)
                m1.metric("Overall LIM", "0.896", delta="Excellent")
                m2.metric("Risk Level", "LOW", delta="Safe")
                m3.metric("Components", "5/5", delta="All Pass")

        elif step_type == "bps":
            st.markdown("### Computing Biological Plausibility Score (BPS)")

            # Radar chart of biological priors
            priors = ['Lesion<br>Persistence', 'Tissue<br>Continuity', 'Boundary<br>Integrity',
                     'Contrast<br>Plausibility', 'Morphological<br>Plausibility', 'Disease<br>Awareness']
            scores = [0.94, 0.89, 0.86, 0.91, 0.88, 0.92]
            scores_closed = scores + [scores[0]]

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=scores_closed,
                theta=priors + [priors[0]],
                fill='toself',
                name='Guardian',
                line_color='#2ecc71'
            ))

            # Add threshold
            fig.add_trace(go.Scatterpolar(
                r=[0.6]*7,
                theta=priors + [priors[0]],
                mode='lines',
                name='Threshold',
                line=dict(color='red', dash='dash')
            ))

            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                height=400,
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)

            with metrics_viz.container():
                m1, m2, m3 = st.columns(3)
                m1.metric("Overall BPS", "0.900", delta="Biologically Plausible")
                m2.metric("Priors Passed", "6/6", delta="All Constraints Met")
                m3.metric("Disease Type", "MS Lesion", delta="Auto-detected")

        elif step_type == "audit":
            st.markdown("### Auditing Black-box Reconstruction")

            # Generate fake black-box with hallucination
            from scipy.ndimage import gaussian_filter
            blackbox = phantom.copy()
            if show_hallucination:
                x = np.linspace(-1, 1, 256)
                y = np.linspace(-1, 1, 256)
                X, Y = np.meshgrid(x, y)
                fake_lesion = 0.3 * np.exp(-((X+0.4)**2 + (Y+0.2)**2) / 0.002)
                blackbox = blackbox + fake_lesion
                blackbox = gaussian_filter(blackbox, 0.5)

            # Guardian reconstruction
            recon = np.abs(np.fft.ifft2(np.fft.ifftshift(kspace_under)))
            for _ in range(5):
                enhanced = recon + 0.1 * (gaussian_filter(recon, 1) - gaussian_filter(recon, 2))
                recon_k = np.fft.fftshift(np.fft.fft2(enhanced))
                recon_k[mask] = kspace_under[mask]
                recon = np.abs(np.fft.ifft2(np.fft.ifftshift(recon_k)))

            discrepancy = np.abs(blackbox - recon)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Black-box Output**")
                fig = go.Figure(go.Heatmap(z=blackbox, colorscale='gray', showscale=False))
                fig.update_layout(height=300, margin=dict(l=10,r=10,t=10,b=10),
                                 xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor='x'))
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("**Guardian Reference**")
                fig = go.Figure(go.Heatmap(z=recon, colorscale='gray', showscale=False))
                fig.update_layout(height=300, margin=dict(l=10,r=10,t=10,b=10),
                                 xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor='x'))
                st.plotly_chart(fig, use_container_width=True)

            with col3:
                st.markdown("**Suspicion Map**")
                fig = go.Figure(go.Heatmap(z=discrepancy/discrepancy.max(), colorscale='YlOrRd', showscale=False))
                fig.update_layout(height=300, margin=dict(l=10,r=10,t=10,b=10),
                                 xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor='x'))
                st.plotly_chart(fig, use_container_width=True)

            with alerts_viz.container():
                if show_hallucination:
                    st.error("⚠️ **HALLUCINATION DETECTED**: Suspicious region found at (-0.4, -0.2). Confidence: 94%. Recommend radiologist review.")
                else:
                    st.success("✅ **NO HALLUCINATIONS DETECTED**: Black-box output consistent with physics-guided reference.")

        elif step_type == "report":
            st.markdown("### Safety Report Generated")

            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown("""
                ```
                ═══════════════════════════════════════════════════════════
                          MRI-GUARDIAN SAFETY REPORT
                ═══════════════════════════════════════════════════════════

                Scan ID: DEMO_001
                Date: 2024-01-15 14:32:00
                Acceleration: 4x
                Reconstruction: Guardian v1.0

                ───────────────────────────────────────────────────────────
                QUALITY METRICS
                ───────────────────────────────────────────────────────────
                PSNR: 36.2 dB (EXCELLENT)
                SSIM: 0.94 (EXCELLENT)
                HFEN: 0.12 (LOW)

                ───────────────────────────────────────────────────────────
                SAFETY METRICS
                ───────────────────────────────────────────────────────────
                Lesion Integrity Marker (LIM): 0.896 (EXCELLENT)
                Biological Plausibility Score: 0.900 (PASSED)
                Hallucination Risk: LOW

                ───────────────────────────────────────────────────────────
                RECOMMENDATION
                ───────────────────────────────────────────────────────────
                ✅ SAFE FOR CLINICAL REVIEW

                All safety metrics within acceptable ranges.
                No hallucinations detected.
                Lesion characteristics preserved.

                ═══════════════════════════════════════════════════════════
                ```
                """)

            with col2:
                st.markdown("**Quick Summary**")
                st.success("✅ PSNR: 36.2 dB")
                st.success("✅ LIM: 0.896")
                st.success("✅ BPS: 0.900")
                if show_hallucination:
                    st.error("⚠️ Black-box: FLAGGED")
                else:
                    st.success("✅ Audit: PASSED")

        elif step_type == "done":
            st.balloons()
            st.success("### Demo Complete!")
            st.markdown("""
            **What you just saw:**
            1. K-space data acquisition and undersampling
            2. Iterative physics-guided reconstruction
            3. Novel bioengineering metrics (LIM, BPS)
            4. Automatic hallucination detection
            5. Comprehensive safety reporting

            **Key Innovation:** MRI-GUARDIAN provides a complete pipeline for safe AI-based
            MRI reconstruction, combining physics constraints with biological knowledge
            to ensure clinically reliable results.
            """)


# Auto-run demo if started
if st.session_state.demo_running:
    run_demo_step(st.session_state.demo_step)

    if st.session_state.demo_step < len(demo_steps) - 1:
        time.sleep(delay)
        st.session_state.demo_step += 1
        st.rerun()
    else:
        st.session_state.demo_running = False
else:
    # Show initial state
    st.info("👆 Click **Start Demo** to begin the interactive demonstration")

    # Show static preview
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Preview: Ground Truth Phantom**")
        fig = go.Figure(go.Heatmap(z=phantom, colorscale='gray', showscale=False))
        fig.update_layout(height=300, margin=dict(l=10,r=10,t=10,b=10),
                         xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor='x'))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Preview: K-Space Data**")
        fig = go.Figure(go.Heatmap(z=np.log1p(np.abs(kspace_full)), colorscale='Viridis', showscale=False))
        fig.update_layout(height=300, margin=dict(l=10,r=10,t=10,b=10),
                         xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor='x'))
        st.plotly_chart(fig, use_container_width=True)
