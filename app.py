from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "fused_data_cleaned.csv"
REPORT_PATH = ROOT / "advanced_analysis_report.md"

SOURCE_LABELS = {
    "京东直采（核心样本）": "jd_direct",
    "全部融合数据": "__all__",
    "搜索引擎补充": "search_supplement",
    "市场估算补充": "estimated",
}

st.set_page_config(page_title="京东多源数据融合分析看板", page_icon="📊", layout="wide")


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    for column in ("price", "comments", "rating", "quality_score"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df["category"] = df["category"].replace("笔记本", "笔记本电脑")
    return df


def normalize_markdown(text):
    """修复分析脚本生成报告时遗留的统一缩进，避免 Markdown 被识别为代码块。"""
    normalized = []
    for line in text.splitlines():
        if line.startswith("        "):
            line = line[8:]
        normalized.append(line.replace("&nbsp;", " "))
    return "\n".join(normalized)


def main():
    df = load_data()
    st.title("京东多源数据融合分析看板")
    st.caption("多品类商品离线解析、清洗、融合与可视化展示")

    st.sidebar.header("数据筛选")
    source_label = st.sidebar.selectbox("数据口径", list(SOURCE_LABELS), index=0)
    source_value = SOURCE_LABELS[source_label]
    scoped = df.copy() if source_value == "__all__" else df[df["source_type"] == source_value].copy()
    categories = ["全部"] + sorted(scoped["category"].dropna().unique().tolist())
    selected_category = st.sidebar.selectbox("品类", categories)
    if selected_category != "全部":
        scoped = scoped[scoped["category"] == selected_category]
    max_comments = int(scoped["comments"].fillna(0).max()) if len(scoped) else 0
    if max_comments > 0:
        min_comments = st.sidebar.slider("最小评论数", 0, max_comments, 0)
    else:
        min_comments = 0
        st.sidebar.caption("当前数据来源没有评论量字段，已跳过评论数筛选。")
    filtered = scoped[scoped["comments"].fillna(0) >= min_comments].copy()

    avg_price = filtered["price"].mean()
    avg_comments = filtered["comments"].mean()
    avg_rating = filtered["rating"].mean()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("样本数", f"{len(filtered)}")
    c2.metric("平均价格", f"¥{avg_price:,.0f}" if pd.notna(avg_price) else "—")
    c3.metric("平均评论", f"{avg_comments:,.0f}" if pd.notna(avg_comments) else "—")
    c4.metric("平均好评率", f"{avg_rating:.1f}%" if pd.notna(avg_rating) else "—")

    tab1, tab2, tab3, tab4 = st.tabs(["概览图表", "品牌/店铺", "样本明细", "实验报告"])
    with tab1:
        if filtered.empty:
            st.warning("当前筛选条件下没有样本。")
        else:
            left, right = st.columns(2)
            with left:
                fig = px.box(filtered, x="category", y="price", color="category", points="all", title="各品类价格分布")
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, width="stretch")
            with right:
                top = filtered.dropna(subset=["comments"]).nlargest(20, "comments")
                if top.empty:
                    st.info("当前数据来源没有评论量，暂不绘制价格—评论分布图。")
                else:
                    scatter_args = dict(data_frame=top, x="price", y="comments", color="category", hover_name="title", title="热门商品价格—评论分布")
                    if top["rating"].notna().any():
                        scatter_args["size"] = "rating"
                    fig = px.scatter(**scatter_args)
                    st.plotly_chart(fig, width="stretch")
            bins = pd.cut(filtered["price"], [-1, 500, 2000, 5000, 10000, 20000, float("inf")], labels=["500以下", "500-2000", "2000-5000", "5000-10000", "10000-20000", "20000以上"])
            segments = filtered.assign(price_segment=bins).groupby(["category", "price_segment"], observed=False, as_index=False).size()
            st.plotly_chart(px.bar(segments, x="price_segment", y="size", color="category", barmode="group", title="价格区间分布"), width="stretch")

    with tab2:
        left, right = st.columns(2)
        with left:
            brands = filtered.assign(brand=filtered["brand"].fillna("未知").replace("", "未知")).groupby("brand", as_index=False).size().nlargest(15, "size")
            fig = px.bar(brands, x="size", y="brand", orientation="h", title="品牌样本量 Top 15")
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, width="stretch")
        with right:
            shops = filtered.groupby("shop", as_index=False).agg(sample_count=("sku", "count"), total_comments=("comments", "sum")).nlargest(15, "total_comments")
            fig = px.bar(shops, x="total_comments", y="shop", orientation="h", color="sample_count", title="店铺竞争格局 Top 15")
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, width="stretch")

    with tab3:
        columns = ["sku", "title", "category", "brand", "shop", "price", "comments", "rating", "quality_score", "source_type", "url"]
        st.dataframe(filtered[[c for c in columns if c in filtered.columns]].reset_index(drop=True), width="stretch", height=520)
        st.download_button("下载当前筛选结果", filtered.to_csv(index=False).encode("utf-8-sig"), "jd_filtered_data.csv", "text/csv")

    with tab4:
        if REPORT_PATH.exists():
            st.caption("以下内容为全部 255 条融合样本的综合分析；顶部指标随侧栏筛选条件变化。")
            st.markdown(normalize_markdown(REPORT_PATH.read_text(encoding="utf-8")))
        else:
            st.info("分析报告文件未部署。")

    st.divider()
    st.caption("北京服装学院人工智能与创新设计学院 · 多源数据融合与分析可视化实训")


if __name__ == "__main__":
    main()
