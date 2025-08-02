import streamlit as st
import asyncio
from client import ApiClient

API_SERVER_URL = "http://127.0.0.1:8000"
client = ApiClient(base_url=API_SERVER_URL)

st.set_page_config(page_title="Fashion Search", layout="wide")
st.title("✨ Visual Fashion Search")

with st.form(key="search_form"):
    search_col, k_col, button_col = st.columns([8, 1, 1])

    with search_col:
        search_query = st.text_input(
            "Search for an item",
            placeholder="e.g., 'blue summer dress with flowers'",
            label_visibility="collapsed",
        )
    with k_col:
        top_k = st.number_input(
            "k",
            min_value=4,
            max_value=40,
            value=12,
            step=4,
            label_visibility="collapsed",
        )
    with button_col:
        search_clicked = st.form_submit_button(label="Search")

st.markdown(
    """
<style>
    .masonry-grid { column-count: 4; column-gap: 1em; }
    .masonry-item {
        position: relative; background-color: #f9f9f9; border-radius: 8px;
        margin-bottom: 1em; display: inline-block; width: 100%;
        overflow: hidden; box-shadow: 0 4px 8px 0 rgba(0,0,0,0.1);
    }
    .masonry-item img { width: 100%; display: block; }
    .score-pill {
        position: absolute; bottom: 8px; right: 8px;
        background-color: rgba(0, 0, 0, 0.7); color: white;
        padding: 4px 10px; border-radius: 12px;
        font-size: 0.8em; font-weight: 500; opacity: 1;
    }
</style>
""",
    unsafe_allow_html=True,
)


if search_clicked and search_query:
    try:
        with st.spinner("Finding the best matches..."):
            response_data = asyncio.run(client.search(query=search_query, top_k=top_k))

        if response_data and response_data.get("results"):
            st.success(f"**Result description:** {response_data.get('summary', 'N/A')}")

            results = sorted(
                response_data["results"],
                key=lambda item: item.get("score", 0),
                reverse=True,
            )

            html_items = []
            for item in results:
                if item.get("image_url"):
                    progress_value = int(item["score"] * 100)
                    html_items.append(
                        f"""
                        <a href="{item["image_url"]}" target="_blank" class="masonry-item">
                            <img src="{item["image_url"]}" alt="Fashion item">
                            <div class="score-pill">{progress_value}% Match</div>
                        </a>
                    """
                    )

            st.html(f"<div class='masonry-grid'>{''.join(html_items)}</div>")

        else:
            st.warning("No results found or an error occurred on the backend.")

    except Exception as e:
        st.error(f"An application error occurred: {e}")
