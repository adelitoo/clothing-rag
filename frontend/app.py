import streamlit as st
from helpers import run_query

st.set_page_config(layout="wide", page_icon="🔍")
st.title("👕 Visual Search Engine")

with st.sidebar:
    st.subheader("🔧 Search Parameters")
    topk = st.number_input(
        "Number of results",
        min_value=1,
        max_value=100,
        value=20,
        step=5,
        help="How many results to return (1-100)",
    )

query = st.text_input(
    "Enter your search query:",
    placeholder="White relaxed shoes...",
    help="Describe what you're looking for",
)

if st.button("🔍 Search Images", type="primary", use_container_width=True) and query:
    with st.status(f"🔍 Finding images matching '{query}'...", expanded=True) as status:
        data = run_query(query, topk)

        if not data or "results" not in data:
            status.update(label="❌ No results found", state="error", expanded=False)
            st.error("Try a different query or check backend services")
            st.stop()

    if summary := data.get("summary"):
        st.markdown("### 📝 Summary of Search Intent")
        st.info(summary)

    cols = st.columns(4)
    for i, item in enumerate(data["results"]):
        if url := item.get("image"):
            with cols[i % 4]:
                with st.container(border=True):
                    st.image(url, use_container_width=True)
                    if item.get("score") is not None:
                        score = float(item["score"])

                        def score_to_color(score):
                            red = int(255 * (1 - score))
                            green = int(255 * score)
                            blue = 180
                            return f"rgb({red},{green},{blue})"

                        color = score_to_color(score)
                        percentage = int(score * 100)

                        st.markdown(
                            f"""
                            <div style="position: relative; height: 16px; margin-top: 3px; margin-bottom: 3px">
                                <div style="
                                    width: 100%;
                                    height: 6px;
                                    background-color: #e0e0e0;
                                    border-radius: 3px;
                                    position: absolute;
                                    top: 50%;
                                    transform: translateY(-50%);
                                ">
                                    <div style="
                                        width: {percentage}%;
                                        height: 100%;
                                        background-color: {color};
                                        border-radius: 3px;
                                    "></div>
                                </div>
                                <div style="
                                    position: absolute;
                                    width: 100%;
                                    text-align: center;
                                    top: 50%;
                                    transform: translateY(-50%);
                                    font-size: 13px;
                                    font-weight: 600;
                                    color: #FFFFFF;
                                    font-family: 'Segoe UI', sans-serif;
                                ">
                                    {percentage}%
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
