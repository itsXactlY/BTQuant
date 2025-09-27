
import os
import re
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import glob

st.set_page_config(
    page_title="BTQuant Quantstats Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def parse_quantstats_html(file_path):
    """Parse a quantstats HTML file and extract key metrics."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
    except Exception as e:
        st.error(f"Error reading file {file_path}: {str(e)}")
        return None

    soup = BeautifulSoup(html_content, 'html.parser')

    filename = os.path.basename(file_path)
    pair_match = re.match(r'([A-Z]+_[A-Z]+)', filename)
    pair = pair_match.group(1) if pair_match else filename.replace('.html', '')

    metrics = {'pair': pair}

    tables = soup.find_all('table')
    if len(tables) > 0:
        main_table = tables[0]
        rows = main_table.find_all('tr')

        for row in rows[1:]:  # Skip header
            cells = row.find_all('td')
            if len(cells) >= 2:
                metric_name = cells[0].get_text(strip=True)
                metric_value = cells[1].get_text(strip=True)

                if '%' in metric_value and metric_value != 'N/A':
                    try:
                        metrics[metric_name] = float(metric_value.replace('%', '')) / 100
                    except ValueError:
                        metrics[metric_name] = None
                else:
                    try:
                        metrics[metric_name] = float(metric_value)
                    except ValueError:
                        metrics[metric_name] = metric_value

    if len(tables) > 1:
        yearly_table = tables[1]
        yearly_returns = {}
        rows = yearly_table.find_all('tr')

        for row in rows[1:]:  # Skip header
            cells = row.find_all('td')
            if len(cells) >= 2:
                year = cells[0].get_text(strip=True)
                return_val = cells[1].get_text(strip=True)
                if '%' in return_val and return_val != 'N/A':
                    try:
                        yearly_returns[year] = float(return_val.replace('%', '')) / 100
                    except ValueError:
                        yearly_returns[year] = None

        metrics['yearly_returns'] = yearly_returns

    return metrics

@st.cache_data
def load_all_reports(folder_path):
    """Load all quantstats HTML reports from a folder."""
    html_files = glob.glob(os.path.join(folder_path, "*.html"))
    all_metrics = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, file_path in enumerate(html_files):
        try:
            metrics = parse_quantstats_html(file_path)
            if metrics:
                all_metrics.append(metrics)
        except Exception as e:
            st.warning(f"Error parsing {file_path}: {str(e)}")

        progress = (i + 1) / len(html_files)
        progress_bar.progress(progress)
        status_text.text(f"Processing {i+1}/{len(html_files)}: {os.path.basename(file_path)}")

    progress_bar.empty()
    status_text.empty()

    return pd.DataFrame(all_metrics)

def create_performance_overview(df):
    """Create overview charts showing key performance metrics."""

    required_cols = ['pair', 'Total Return', 'Sharpe', 'Max Drawdown', 'Volatility (ann.)']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        return None

    clean_df = df.dropna(subset=['Total Return', 'Sharpe', 'Max Drawdown', 'Volatility (ann.)'])

    if clean_df.empty:
        st.warning("No valid data available for overview charts")
        return None

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Volatility (Annual)'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )

    fig.add_trace(
        go.Bar(
            x=clean_df['pair'],
            y=clean_df['Total Return'] * 100,  # Convert to percentage
            name='Total Return (%)',
            marker_color='lightblue'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(
            x=clean_df['pair'],
            y=clean_df['Sharpe'],
            name='Sharpe Ratio',
            marker_color='lightgreen'
        ),
        row=1, col=2
    )

    fig.add_trace(
        go.Bar(
            x=clean_df['pair'],
            y=clean_df['Max Drawdown'] * 100,  # Convert to percentage
            name='Max Drawdown (%)',
            marker_color='lightcoral'
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Bar(
            x=clean_df['pair'],
            y=clean_df['Volatility (ann.)'] * 100,  # Convert to percentage
            name='Volatility (%)',
            marker_color='lightyellow'
        ),
        row=2, col=2
    )

    fig.update_layout(
        height=600,
        showlegend=False,
        title_text="Performance Metrics Comparison"
    )

    return fig

def create_risk_return_scatter(df):
    """Create risk-return scatter plot with fixed size handling."""

    required_cols = ['Volatility (ann.)', 'Total Return', 'Sharpe', 'Max Drawdown', 'pair']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        st.error(f"Missing required columns for risk-return analysis: {missing_cols}")
        return None

    plot_df = df.dropna(subset=['Volatility (ann.)', 'Total Return', 'Sharpe']).copy()

    if plot_df.empty:
        st.warning("No valid data available for risk-return plot")
        return None

    fig = go.Figure()

    # Fix the size issue: Transform Sharpe ratio to always positive values for marker size
    # Scale to range [5, 50] for good visibility
    min_sharpe = plot_df['Sharpe'].min()
    max_sharpe = plot_df['Sharpe'].max()

    if max_sharpe == min_sharpe:
        marker_sizes = [15] * len(plot_df)
    else:
        normalized_sharpe = (plot_df['Sharpe'] - min_sharpe) / (max_sharpe - min_sharpe)
        marker_sizes = 5 + normalized_sharpe * 45

    fig.add_trace(go.Scatter(
        x=plot_df['Volatility (ann.)'] * 100,
        y=plot_df['Total Return'] * 100,
        mode='markers+text',
        text=plot_df['pair'],
        textposition="top center",
        marker=dict(
            size=marker_sizes,
            color=plot_df['Max Drawdown'] * 100,  # Color by drawdown %
            colorscale='RdYlBu_r',  # Reverse so red = bad (high drawdown)
            colorbar=dict(title="Max Drawdown (%)"),
            line=dict(width=1, color='white'),
            opacity=0.7
        ),
        hovertemplate=
        '<b>%{text}</b><br>' +
        'Volatility: %{x:.1f}%<br>' +
        'Total Return: %{y:.1f}%<br>' +
        'Sharpe: %{customdata[0]:.3f}<br>' +
        'Max Drawdown: %{customdata[1]:.1f}%<br>' +
        '<extra></extra>',
        customdata=np.column_stack([
            plot_df['Sharpe'], 
            plot_df['Max Drawdown'] * 100
        ])
    ))

    fig.update_layout(
        title='Risk-Return Analysis (Bubble size based on Sharpe Ratio)',
        xaxis_title='Volatility (Annual %)',
        yaxis_title='Total Return (%)',
        height=500,
        hovermode='closest'
    )

    return fig

def create_performance_ranking(df):
    """Create performance ranking table with proper error handling."""

    required_cols = ['pair', 'Total Return', 'Sharpe', 'Max Drawdown']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        st.error(f"Missing required columns for ranking: {missing_cols}")
        return None

    clean_df = df.dropna(subset=required_cols).copy()

    if clean_df.empty:
        st.warning("No valid data available for ranking")
        return None

    clean_df['Risk_Adjusted_Return'] = clean_df['Total Return'] / np.abs(clean_df['Max Drawdown'])

    clean_df['Risk_Adjusted_Return'] = clean_df['Risk_Adjusted_Return'].replace([np.inf, -np.inf], np.nan)
    clean_df['Risk_Adjusted_Return'] = clean_df['Risk_Adjusted_Return'].fillna(0)

    clean_df['Composite_Score'] = (
        clean_df['Sharpe'].rank(ascending=False, na_option='bottom') +
        clean_df['Total Return'].rank(ascending=False, na_option='bottom') +
        clean_df['Max Drawdown'].rank(ascending=True, na_option='bottom') +  # Lower drawdown is better
        clean_df['Risk_Adjusted_Return'].rank(ascending=False, na_option='bottom')
    ) / 4

    ranked_df = clean_df.sort_values('Composite_Score').copy()

    display_cols = ['pair', 'Total Return', 'Sharpe', 'Max Drawdown', 'Composite_Score']

    if 'CAGR% (Annual Return)' in clean_df.columns:
        display_cols.insert(2, 'CAGR% (Annual Return)')

    if 'Volatility (ann.)' in clean_df.columns:
        display_cols.insert(-1, 'Volatility (ann.)')

    ranking_table = ranked_df[display_cols].copy()

    pct_cols = ['Total Return', 'Max Drawdown']
    if 'CAGR% (Annual Return)' in ranking_table.columns:
        pct_cols.append('CAGR% (Annual Return)')
    if 'Volatility (ann.)' in ranking_table.columns:
        pct_cols.append('Volatility (ann.)')

    for col in pct_cols:
        if col in ranking_table.columns:
            ranking_table[col] = ranking_table[col].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A")

    if 'Sharpe' in ranking_table.columns:
        ranking_table['Sharpe'] = ranking_table['Sharpe'].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")

    ranking_table['Composite_Score'] = ranking_table['Composite_Score'].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")

    return ranking_table

def create_yearly_performance_heatmap(df):
    """Create yearly performance heatmap with error handling."""

    if 'yearly_returns' not in df.columns:
        st.warning("No yearly returns data available for heatmap")
        return None

    all_years = set()
    valid_pairs = []

    for idx, row in df.iterrows():
        yearly_returns = row.get('yearly_returns', {})
        if isinstance(yearly_returns, dict) and yearly_returns:
            all_years.update(yearly_returns.keys())
            valid_pairs.append(idx)

    if not all_years or not valid_pairs:
        st.warning("No valid yearly returns data found")
        return None

    all_years = sorted(list(all_years))

    heatmap_data = []
    pair_names = []

    for idx in valid_pairs:
        row = df.loc[idx]
        yearly_returns = row.get('yearly_returns', {})
        year_data = []

        for year in all_years:
            if year in yearly_returns and yearly_returns[year] is not None:
                year_data.append(yearly_returns[year] * 100)
            else:
                year_data.append(None)

        if any(x is not None for x in year_data):
            heatmap_data.append(year_data)
            pair_names.append(row['pair'])

    if not heatmap_data:
        st.warning("No valid yearly returns data to display")
        return None

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=all_years,
        y=pair_names,
        colorscale='RdYlGn',
        text=[[f"{val:.1f}%" if val is not None else "N/A" for val in row] for row in heatmap_data],
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False,
        zmid=0  # Center colorscale at 0%
    ))

    fig.update_layout(
        title='Yearly Returns Heatmap (%)',
        xaxis_title='Year',
        yaxis_title='Trading Pair',
        height=max(400, len(pair_names) * 30)
    )

    return fig

def main():
    st.title("BTQuant Trading Pair Analysis Dashboard")
    st.markdown("**Comprehensive analysis of multiple trading pairs from quantstats reports**")

    st.sidebar.header("📁 Data Input")

    # Option 1: Upload multiple files
    uploaded_files = st.sidebar.file_uploader(
        "Upload Quantstats HTML files",
        type=['html'],
        accept_multiple_files=True,
        help="Upload multiple quantstats HTML reports for comparison"
    )

    # Option 2: Specify folder path
    folder_path = st.sidebar.text_input(
        "Or specify folder path",
        placeholder="/path/to/quantstats/reports/",
        help="Path to folder containing quantstats HTML files"
    )

    df = pd.DataFrame()

    if uploaded_files:
        st.sidebar.success(f"Loaded {len(uploaded_files)} files")
        all_metrics = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, uploaded_file in enumerate(uploaded_files):
            temp_path = f"temp_{uploaded_file.name}"
            try:
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                metrics = parse_quantstats_html(temp_path)
                if metrics:
                    all_metrics.append(metrics)

                progress = (i + 1) / len(uploaded_files)
                progress_bar.progress(progress)
                status_text.text(f"Processing {i+1}/{len(uploaded_files)}: {uploaded_file.name}")

            except Exception as e:
                st.warning(f"Error processing {uploaded_file.name}: {str(e)}")
            finally:
                try:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except:
                    pass

        progress_bar.empty()
        status_text.empty()

        if all_metrics:
            df = pd.DataFrame(all_metrics)
            st.sidebar.success(f"Successfully processed {len(df)} pairs")

    elif folder_path and os.path.exists(folder_path):
        with st.spinner("Loading reports from folder..."):
            df = load_all_reports(folder_path)
        if not df.empty:
            st.sidebar.success(f"Loaded {len(df)} reports from folder")

    elif not uploaded_files and not folder_path:
        st.info("👆 Please upload HTML files or specify a folder path to begin analysis")
        return

    if df.empty:
        st.warning("No valid quantstats reports found. Please check your files or folder path.")
        return

    # Main dashboard
    st.header("📊 Performance Overview")

    try:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if 'Total Return' in df.columns and not df['Total Return'].isna().all():
                best_performer = df.loc[df['Total Return'].idxmax()]
                st.metric(
                    "🥇 Best Total Return",
                    f"{best_performer['pair']}",
                    f"{best_performer['Total Return']*100:.2f}%"
                )
            else:
                st.metric("🥇 Best Total Return", "N/A", "No data")

        with col2:
            if 'Sharpe' in df.columns and not df['Sharpe'].isna().all():
                best_sharpe = df.loc[df['Sharpe'].idxmax()]
                st.metric(
                    "📈 Best Sharpe Ratio",
                    f"{best_sharpe['pair']}",
                    f"{best_sharpe['Sharpe']:.3f}"
                )
            else:
                st.metric("📈 Best Sharpe Ratio", "N/A", "No data")

        with col3:
            if 'Max Drawdown' in df.columns and not df['Max Drawdown'].isna().all():
                best_drawdown = df.loc[df['Max Drawdown'].idxmax()]  # Closest to 0
                st.metric(
                    "🛡️ Lowest Drawdown",
                    f"{best_drawdown['pair']}",
                    f"{best_drawdown['Max Drawdown']*100:.2f}%"
                )
            else:
                st.metric("🛡️ Lowest Drawdown", "N/A", "No data")

        with col4:
            if 'Volatility (ann.)' in df.columns and not df['Volatility (ann.)'].isna().all():
                lowest_vol = df.loc[df['Volatility (ann.)'].idxmin()]
                st.metric(
                    "⚖️ Lowest Volatility",
                    f"{lowest_vol['pair']}",
                    f"{lowest_vol['Volatility (ann.)']*100:.2f}%"
                )
            else:
                st.metric("⚖️ Lowest Volatility", "N/A", "No data")

    except Exception as e:
        st.error(f"Error displaying metrics cards: {str(e)}")

    st.subheader("Performance Metrics Comparison")
    try:
        perf_fig = create_performance_overview(df)
        if perf_fig:
            st.plotly_chart(perf_fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating performance overview: {str(e)}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk-Return Profile")
        try:
            risk_return_fig = create_risk_return_scatter(df)
            if risk_return_fig:
                st.plotly_chart(risk_return_fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating risk-return plot: {str(e)}")

    with col2:
        st.subheader("Performance Ranking")
        try:
            ranking_table = create_performance_ranking(df)
            if ranking_table is not None:
                st.dataframe(
                    ranking_table,
                    use_container_width=True,
                    height=400
                )
        except Exception as e:
            st.error(f"Error creating ranking table: {str(e)}")

    st.subheader("Yearly Performance Heatmap")
    try:
        yearly_fig = create_yearly_performance_heatmap(df)
        if yearly_fig:
            st.plotly_chart(yearly_fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating yearly heatmap: {str(e)}")

    st.subheader("📋 Detailed Metrics")

    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            default_cols = [col for col in ['Total Return', 'CAGR% (Annual Return)', 'Sharpe', 'Max Drawdown', 'Volatility (ann.)'] if col in numeric_cols]
            display_cols = st.multiselect(
                "Select metrics to display:",
                options=numeric_cols,
                default=default_cols[:5] if default_cols else numeric_cols[:5]
            )

            if display_cols:
                detail_df = df[['pair'] + display_cols].copy()

                pct_cols = [col for col in display_cols if any(keyword in col.lower() for keyword in ['return', 'drawdown', 'volatility'])]
                for col in pct_cols:
                    detail_df[col] = detail_df[col].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A")

                st.dataframe(detail_df, use_container_width=True)
        else:
            st.warning("No numeric columns found for detailed metrics")
    except Exception as e:
        st.error(f"Error creating detailed metrics table: {str(e)}")

    st.subheader("💾 Export Data")
    '''
    TODO: Add MSsQL export functionality
    '''
    try:
        if st.button("Generate CSV Export"):
            csv_data = df.select_dtypes(include=[np.number, 'object']).copy()

            if 'yearly_returns' in csv_data.columns:
                csv_data = csv_data.drop('yearly_returns', axis=1)

            csv_string = csv_data.to_csv(index=False)

            st.download_button(
                label="📥 Download CSV",
                data=csv_string,
                file_name="trading_pairs_analysis.csv",
                mime="text/csv"
            )
        if st.button("Export to MS SQL Database"):
            st.info("MS SQL export functionality is not yet implemented.")
    except Exception as e:
        st.error(f"Error generating CSV export: {str(e)}")

if __name__ == "__main__":
    main()
