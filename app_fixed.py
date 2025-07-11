import streamlit as st
import pandas as pd
import os
import altair as alt
from itertools import product
import inflect

p = inflect.engine()

# -------------------------
# 工具函数
# -------------------------
def normalize(text):
    text = text.lower().strip()
    return p.singular_noun(text) or text

def parse_keywords(text):
    if pd.isna(text) or not str(text).strip():
        return []
    return [normalize(x) for x in str(text).split("/") if x.strip()]

def load_term_library(path="term_library.xlsx"):
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()
    term_dict = {row["分类标签"]: parse_keywords(row["对应词"]) for _, row in df.iterrows()}
    return term_dict, df

def load_spu_rules(path="spu_rules.xlsx"):
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()
    return df

def expand_keyword_group(group, term_dict):
    parts = group.split("+")
    expanded = []
    for p in parts:
        if p in term_dict:
            expanded.append(term_dict[p])
        else:
            expanded.append([normalize(p)])
    return [" ".join(x) for x in product(*expanded)]

def match_grouped_terms(term, group_str, term_dict):
    words = term.split()
    phrases = expand_keyword_group(group_str, term_dict)
    return any(all(w in words for w in phrase.split()) for phrase in phrases)

def match_any_term(term, keywords, term_dict):
    words = term.split()
    expanded = []
    for k in keywords:
        if k in term_dict:
            expanded.extend(term_dict[k])
        else:
            expanded.append(normalize(k))
    return any(k in words for k in expanded)

def classify_term(search_term, rules_row, term_dict):
    term_raw = str(search_term).lower()
    words = [normalize(w) for w in term_raw.split()]
    term_norm = " ".join(words)

    hit_level = "未分类"
    hit_keywords = ""
    hit_negation = ""

    # 否定匹配函数
    def has_neg(field, label):
        if isinstance(field, str) and field.strip():
            for word in parse_keywords(field):
                if word in words:
                    return label + ":" + word
        return ""

    # 四级
    if not has_neg(rules_row.get("否定词根_四级", ""), "四级"):
        for group in parse_keywords(rules_row.get("四级", "")):
            if match_grouped_terms(term_norm, group, term_dict):
                return 1, 0, 0, "四类", group, ""
    else:
        hit_negation = has_neg(rules_row.get("否定词根_四级", ""), "四级")

    # 三级
    if not has_neg(rules_row.get("否定词根_三级", ""), "三级"):
        for group in parse_keywords(rules_row.get("三级", "")):
            if match_grouped_terms(term_norm, group, term_dict):
                return 0, 1, 0, "三类", group, ""
    else:
        hit_negation = has_neg(rules_row.get("否定词根_三级", ""), "三级")

    # 二级
    if not has_neg(rules_row.get("否定词根_二级", ""), "二级"):
        for token in parse_keywords(rules_row.get("二级", "")):
            if match_any_term(term_norm, [token], term_dict):
                return 0, 0, 1, "二类", token, ""
    else:
        hit_negation = has_neg(rules_row.get("否定词根_二级", ""), "二级")

    return 0, 0, 0, "未分类", hit_keywords, hit_negation

def classify_all(search_df, mapping_df, rules_df, term_dict):
    merged = search_df.merge(mapping_df, on="ASIN", how="left")
    merged = merged.merge(rules_df, on="SPU", how="left")

    if "SPU运营_x" in merged.columns:
        merged = merged.rename(columns={"SPU运营_x": "SPU运营"})
    elif "SPU运营_y" in merged.columns:
        merged = merged.rename(columns={"SPU运营_y": "SPU运营"})

    merged["月份"] = pd.to_datetime(merged["日期"], errors='coerce').dt.to_period("M").astype(str)

    merged[["二级匹配", "三级匹配", "四级匹配", "分类级别", "命中关键词", "命中否定词根"]] = merged.apply(
        lambda row: pd.Series(classify_term(row.get("搜索词", ""), row, term_dict)), axis=1
    )
    return merged

def group_by_operator(df):
    grouped = df.groupby(["月份","SPU运营", "分类级别"]).agg(
        曝光总量=("搜索漏斗 - 展示量: 总数", "sum"),
        曝光ASIN数=("搜索漏斗 - 展示量: ASIN 计数", "sum"),
        购买总量=("搜索漏斗 - 购买次数: 总数", "sum"),
        购买ASIN数=("搜索漏斗 - 购买次数: ASIN 计数", "sum")
    ).reset_index()
    # 转换为数值类型，避免字符串类型出错
    for col in ["曝光总量", "曝光ASIN数", "购买总量", "购买ASIN数"]:
        grouped[col] = pd.to_numeric(grouped[col], errors="coerce").fillna(0)
    grouped["曝光份额占比"] = grouped["曝光ASIN数"] / grouped["曝光总量"]
    grouped["购买份额占比"] = grouped["购买ASIN数"] / grouped["购买总量"]
    return grouped

def plot_trend_exposure_share(df, title="曝光份额趋势图"):
    g = df.groupby(["月份", "分类级别"]).agg(
        曝光总量=("搜索漏斗 - 展示量: 总数", "sum"),
        曝光ASIN数=("搜索漏斗 - 展示量: ASIN 计数", "sum")
    ).reset_index()
    g["份额"] = g["曝光ASIN数"] / g["曝光总量"]

    chart = alt.Chart(g).mark_line(point=True).encode(
        x="月份:N",
        y=alt.Y("份额:Q", axis=alt.Axis(format=".0%")),
        color="分类级别:N",
        tooltip=["月份", "分类级别", alt.Tooltip("份额", format=".2%")]
    ).properties(title=title, width=800)

    return chart
def plot_trend_purchase_share(df, title="购买份额趋势图"):
    g = df.groupby(["月份", "分类级别"]).agg(
        购买总量=("搜索漏斗 - 购买次数: 总数", "sum"),
        购买ASIN数=("搜索漏斗 - 购买次数: ASIN 计数", "sum")
    ).reset_index()
    g["份额"] = g["购买ASIN数"] / g["购买总量"]

    chart = alt.Chart(g).mark_line(point=True).encode(
        x="月份:N",
        y=alt.Y("份额:Q", axis=alt.Axis(format=".0%")),
        color="分类级别:N",
        tooltip=["月份", "分类级别", alt.Tooltip("份额", format=".2%")]
    ).properties(title=title, width=800)

    return chart
# 🔌 Supabase连接
from supabase import create_client, Client

url = "https://texekgedycnthkeivbtw.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InRleGVrZ2VkeWNudGhrZWl2YnR3Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1MjIxMjU5MCwiZXhwIjoyMDY3Nzg4NTkwfQ.tcTI53XXVt7lt4AdloHHoI89Ig7zFxi50MUqagKq_Oo"
supabase: Client = create_client(url, key)

# ✅ 缓存 Supabase 加载
@st.cache_data(ttl=1800)
def load_supabase_data_paged(max_pages=100, page_size=5000):
    all_data = []
    for i in range(max_pages):
        start = i * page_size
        end = start + page_size - 1
        response = supabase.table("search_terms_row").select("*").range(start, end).execute()
        page_data = response.data
        if not page_data:
            break  # 没有更多数据
        all_data.extend(page_data)
    df = pd.DataFrame(all_data)
    df.columns = df.columns.str.strip()
    df["日期"] = pd.to_datetime(df["日期"], errors='coerce')
    return df


def load_sku_mapping():
    response = supabase.table("sku_mapping").select("*").execute()
    df = pd.DataFrame(response.data)
    df.columns = df.columns.str.strip()
    return df.drop_duplicates(subset="ASIN")

def load_spu_rules():
    response = supabase.table("spu_rules").select("*").execute()
    df = pd.DataFrame(response.data)
    df.columns = df.columns.str.strip()
    return df

def load_term_library():
    response = supabase.table("term_library").select("*").execute()
    df = pd.DataFrame(response.data)
    df.columns = df.columns.str.strip()
    term_dict = {row["分类标签"]: parse_keywords(row["对应词"]) for _, row in df.iterrows()}
    return term_dict, df

# -------------------------
# 页面布局
# -------------------------
st.set_page_config("搜索词分类表现分析", layout="wide")
st.title("📊 搜索词分类表现分析工具")

with st.spinner("从 Supabase 分页加载搜索词数据..."):
    df = load_supabase_data_paged(max_pages=100, page_size=5000)  # 最多加载 50 万条
    mapping_df = load_sku_mapping()
    mapping_df = mapping_df.drop_duplicates(subset="ASIN")
    rules_df = load_spu_rules()
    term_dict, term_df = load_term_library()
    merged = classify_all(df, mapping_df, rules_df, term_dict)

# 🧾 词库维护
st.subheader("🧾 词库维护（term_library.xlsx）")
term_df_edit = st.data_editor(term_df, num_rows="dynamic", use_container_width=True)
col1, col2 = st.columns(2)
with col1:
    if st.button("💾 保存词库"):
    data = term_df_edit.to_dict(orient="records")
    supabase.table("term_library").delete().neq("分类标签", "").execute()  # 清空
    supabase.table("term_library").insert(data).execute()
    st.success("已保存到 Supabase！请刷新页面生效。")
with col2:
    st.download_button("📤 导出词库", term_df_edit.to_csv(index=False).encode("utf-8-sig"), file_name="term_library.csv")

# 📘 分类规则维护
st.subheader("📘 分类规则维护（spu_rules.xlsx）")
rules_edit = st.data_editor(rules_df, use_container_width=True, num_rows="dynamic")
col3, col4 = st.columns(2)
with col3:
    if st.button("💾 保存分类规则"):
    data = rules_edit.to_dict(orient="records")
    supabase.table("spu_rules").delete().neq("SPU", "").execute()  # 清空原表
    supabase.table("spu_rules").insert(data).execute()
    st.success("分类规则已保存到 Supabase，请刷新页面加载新规则。")
with col4:
    st.download_button("📥 下载分类规则", rules_edit.to_csv(index=False).encode("utf-8-sig"), file_name="spu_rules.csv")

# 🔍 SPU 匹配词查看
st.subheader("🔍 查看 SPU 匹配词")
selected_spu = st.selectbox("选择要查看的 SPU：", rules_df["SPU"].dropna().unique())
spu_row = rules_df[rules_df["SPU"] == selected_spu].iloc[0]
with st.expander(f"📂 SPU: {selected_spu} 的分类规则"):
    st.write("**二级关键词：**", parse_keywords(spu_row.get("二级", "")))
    st.write("**三级关键词组合：**", parse_keywords(spu_row.get("三级", "")))
    st.write("**四级关键词组合：**", parse_keywords(spu_row.get("四级", "")))
    st.write("**否定词根_二级：**", parse_keywords(spu_row.get("否定词根_二级", "")))
    st.write("**否定词根_三级：**", parse_keywords(spu_row.get("否定词根_三级", "")))
    st.write("**否定词根_四级：**", parse_keywords(spu_row.get("否定词根_四级", "")))

# 🎯 筛选条件
st.subheader("🔍 筛选条件")
levels = ["四类", "三类", "二类", "未分类"]
filters = {
    "分类级别": st.multiselect("分类级别", levels, default=levels[:-1]),
    "月份": st.multiselect("月份", merged["月份"].dropna().unique().tolist()),
    "SPU运营": st.multiselect("SPU运营", merged["SPU运营"].dropna().unique().tolist()),
    "SPU": st.multiselect("SPU", merged["SPU"].dropna().unique().tolist()),
    "板块": st.multiselect("板块", merged["板块"].dropna().unique().tolist()),
    "产品线": st.multiselect("产品线", merged["产品线"].dropna().unique().tolist()),
}
kw_input = st.text_input("搜索词包含：")

mask = merged["分类级别"].isin(filters["分类级别"])
for col, val in filters.items():
    if val and col != "分类级别":
        mask &= merged[col].isin(val)
if kw_input:
    mask &= merged["搜索词"].str.contains(kw_input, na=False, case=False)

filtered = merged[mask]

st.info(f"✅ 当前加载数据量：{len(df)} 条")
st.dataframe(df['日期'].dt.to_period("M").value_counts().sort_index().rename("条数").reset_index().rename(columns={"index": "月份"}))
# 📈 汇总与明细
st.subheader("📈 各类词表现汇总")
st.dataframe(group_by_operator(filtered), use_container_width=True)

st.subheader("📋 筛选后的明细数据")
st.dataframe(filtered[["搜索词", "分类级别", "命中关键词", "命中否定词根", "SPU", "SPU运营", "板块", "产品线","月份"] +
                      [col for col in filtered.columns if col.startswith("搜索漏斗")]],
             use_container_width=True)
st.download_button("📥 下载筛选明细", filtered.to_csv(index=False).encode("utf-8-sig"), file_name="筛选搜索词明细.csv")

# 📊 趋势图
st.subheader("📊 曝光份额趋势图")
st.altair_chart(plot_trend_exposure_share(filtered), use_container_width=True)

st.subheader("🛒 购买份额趋势图")
st.altair_chart(plot_trend_purchase_share(filtered), use_container_width=True)

