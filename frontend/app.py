import streamlit as st
import asyncio
from client import ApiClient
import uuid

API_SERVER_URL = "http://127.0.0.1:8000"

CATEGORY_DISPLAY_NAMES = {
    "shirt": "Tops & Shirts",
    "jacket": "Jackets & Coats",
    "pants": "Pants & Trousers",
    "shoes": "Shoes",
    "dress": "Dresses & Jumpsuits",
    "handbag": "Bags & Handbags",
    "tie": "Accessories",
    "belt": "Accessories",
    "accessories": "Accessories",
    "jewelry": "Jewelry",
    "swimwear": "Swimwear",
    "Garment Upper body": "Tops & Jackets",
    "Garment Lower body": "Pants & Skirts",
    "Garment Full body": "Dresses & Jumpsuits",
    "Underwear": "Underwear & Socks",
}

try:
    client = ApiClient(base_url=API_SERVER_URL)
except Exception as e:
    st.error(f"Failed to initialize API Client: {e}")
    st.stop()

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    print(f"New session started: {st.session_state.session_id}")

st.set_page_config(page_title="Visual Fashion Search", layout="wide")
st.title("✨ Visual Fashion Search")

with st.form(key="search_form"):
    search_col, buttons_col = st.columns([6, 1])

    with search_col:
        search_query = st.text_input(
            "Search for an item or describe an outfit",
            placeholder="e.g., elegant outfit for dinner",
            label_visibility="collapsed",
        )

    with buttons_col:
        search_button_col, reset_button_col = st.columns([1, 1], gap="small")

        with search_button_col:
            search_clicked = st.form_submit_button(
                label="Search",
                type="primary",
                use_container_width=True,
                help="Run a new search",
            )

        with reset_button_col:
            reset_clicked = st.form_submit_button(
                label="Clear",
                use_container_width=True,
                help="Reset the conversation history",
            )

st.markdown(
    """
<style>
    .masonry-grid {
        column-count: 4;
        column-gap: 1em;
    }
    .masonry-item {
        position: relative;
        background-color: #f9f9f9;
        border-radius: 8px;
        margin-bottom: 1em;
        display: inline-block;
        width: 100%;
        overflow: hidden;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.1);
    }
    .masonry-item img {
        width: 100%;
        display: block;
    }
    .score-pill {
        position: absolute;
        bottom: 8px;
        right: 8px;
        background-color: rgba(0, 0, 0, 0.7);
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: 500;
    }
</style>
""",
    unsafe_allow_html=True,
)

if reset_clicked:
    st.session_state.session_id = str(uuid.uuid4())
    st.toast("Conversation has been reset!", icon="🧹")
    print(f"🔄 Conversation Reset. New Session ID: {st.session_state.session_id}")

elif search_clicked and search_query:
    try:
        with st.spinner("Finding the best matches for your outfit..."):
            response_data = asyncio.run(
                client.agent_recommend(
                    query=search_query, session_id=st.session_state.session_id
                )
            )

        if response_data and response_data.get("categorized_articles"):
            st.markdown(response_data.get("summary_text", ""))

            categorized_articles = response_data.get("categorized_articles", {})

            for category, articles in categorized_articles.items():
                display_title = CATEGORY_DISPLAY_NAMES.get(
                    category, category.replace("_", " ").capitalize()
                )
                st.subheader(display_title)

                if not articles:
                    st.info("No items were found for this category.")
                    st.markdown("---")
                    continue

                valid_articles_for_display = []
                for item in articles:
                    if item and item.get("image_url"):
                        valid_articles_for_display.append(
                            {
                                "article_id": item.get("article_id", "unknown"),
                                "image_url": item.get("image_url"),
                                "relevance_score": item.get("relevance_score", 0.0),
                            }
                        )

                sorted_articles = sorted(
                    valid_articles_for_display,
                    key=lambda x: x["relevance_score"],
                    reverse=True,
                )

                if sorted_articles:
                    html_items = []
                    for item in sorted_articles:
                        progress_value = int(item["relevance_score"] * 100)
                        html_items.append(
                            f"""
                            <a href="{item["image_url"]}" target="_blank" class="masonry-item">
                                <img src="{item["image_url"]}" alt="Fashion item {item["article_id"]}">
                                <div class="score-pill">{progress_value}% Match</div>
                            </a>
                            """
                        )
                    st.html(f"<div class='masonry-grid'>{''.join(html_items)}</div>")
                else:
                    st.info("No items with images could be found for this category.")

                st.markdown("---")

        else:
            error_message = response_data.get(
                "error", "No items found. Please try a different search."
            )
            st.warning(error_message)

    except Exception as e:
        st.error(f"An application error occurred: {e}")
        import traceback

        traceback.print_exc()
