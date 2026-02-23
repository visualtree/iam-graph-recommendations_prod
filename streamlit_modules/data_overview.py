"""
Data overview and statistics display for Streamlit application
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from .metrics_calculator import get_data_complexity_metrics, calculate_real_processing_metrics

def display_data_overview(graph_dfs):
    """Display comprehensive data overview with real statistics"""
    
    st.markdown("### 📊 Your Data Overview")
    
    # Calculate real statistics
    complexity_metrics = get_data_complexity_metrics()
    processing_metrics = calculate_real_processing_metrics()
    
    if not complexity_metrics:
        st.error("Unable to calculate data metrics")
        return
    
    # Main metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        active_pct = (complexity_metrics['active_users'] / complexity_metrics['total_users']) * 100
        st.metric(
            "Users", 
            f"{complexity_metrics['total_users']:,}", 
            delta=f"{complexity_metrics['active_users']:,} active ({active_pct:.0f}%)",
            help="Active users with access relationships"
        )
    
    with col2:
        st.metric(
            "Entitlements", 
            f"{complexity_metrics['total_entitlements']:,}",
            delta=f"Avg {complexity_metrics['avg_access_per_user']:.0f} per user",
            help="Total access permissions in the system"
        )
    
    with col3:
        st.metric(
            "Access Relations", 
            f"{complexity_metrics['total_relationships']:,}",
            delta=f"Density: {complexity_metrics['graph_density']:.1%}",
            help="User-to-entitlement relationships in your graph"
        )
    
    with col4:
        org_complexity = complexity_metrics['organizational_complexity']
        total_systems = org_complexity['systems']
        
        if total_systems > 0:
            avg_ents_per_system = complexity_metrics['total_entitlements'] / total_systems
            st.metric(
                "Systems", 
                f"{total_systems:,}",
                delta=f"Avg {avg_ents_per_system:.0f} ents/system",
                help="Endpoint systems providing entitlements"
            )
        else:
            st.metric("Systems", "N/A", help="System data not available")
    
    # Detailed insights section
    display_data_insights(complexity_metrics, processing_metrics)
    
    # Visual analytics
    display_data_visualizations(graph_dfs, complexity_metrics)

def display_data_insights(complexity_metrics, processing_metrics):
    """Display detailed data insights"""
    
    st.markdown("#### 🔍 Data Insights")
    
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        st.markdown("**📈 Organizational Structure:**")
        
        graph_density = complexity_metrics['graph_density']
        complexity_category = complexity_metrics['complexity_category']
        
        density_assessment = (
            "sparse" if graph_density < 0.1 else 
            "moderate" if graph_density < 0.3 else 
            "dense"
        )
        
        org_complexity = complexity_metrics['organizational_complexity']
        
        st.markdown(f"""
        - **Complexity Category**: {complexity_category}
        - **Graph Density**: {graph_density:.1%} ({density_assessment} access patterns)
        - **Average Access**: {complexity_metrics['avg_access_per_user']:.0f} entitlements per user
        - **Organizations**: {org_complexity['organizations']} departments
        - **Roles**: {org_complexity['roles']} distinct roles
        """)
    
    with insight_col2:
        if processing_metrics:
            st.markdown("**⚡ Processing Implications:**")
            
            st.markdown(f"""
            - **ML Processing**: {processing_metrics['ml_total']:.1f} seconds estimated
            - **SQL Alternative**: {processing_metrics['sql_total']:.0f} seconds estimated
            - **Speed Advantage**: {processing_metrics['sql_total']//processing_metrics['ml_total']:.0f}x faster with ML
            - **Data Complexity**: {processing_metrics['data_factor']:.1f}x scaling factor
            - **Optimization**: Suitable for ML approach
            """)

def display_data_visualizations(graph_dfs, complexity_metrics):
    """Display data visualizations"""
    
    st.markdown("#### 📊 Data Distribution Analysis")
    
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        # Access distribution histogram
        access_distribution = graph_dfs['entrecon'].groupby('UserId').size()
        
        fig = px.histogram(
            x=access_distribution.values,
            nbins=20,
            title="Access Distribution per User",
            labels={'x': 'Number of Entitlements', 'y': 'Number of Users'}
        )
        
        fig.update_layout(
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Access statistics
        st.markdown("**Access Statistics:**")
        st.markdown(f"- **Median**: {access_distribution.median():.0f} entitlements")
        st.markdown(f"- **Mean**: {access_distribution.mean():.1f} entitlements")
        st.markdown(f"- **Range**: {access_distribution.min()}-{access_distribution.max()}")
        
        # Power users
        power_user_threshold = access_distribution.quantile(0.95)
        power_users = len(access_distribution[access_distribution >= power_user_threshold])
        st.markdown(f"- **Power Users**: {power_users} ({power_users/len(access_distribution)*100:.1f}%)")
    
    with viz_col2:
        # Organizational distribution
        if 'NOrganisationId' in graph_dfs['users'].columns:
            org_distribution = graph_dfs['users'].groupby('NOrganisationId').size().sort_values(ascending=False)
            
            # Top 10 organizations
            top_orgs = org_distribution.head(10)
            
            fig = px.bar(
                x=top_orgs.index.astype(str),
                y=top_orgs.values,
                title="Top 10 Organizations by User Count",
                labels={'x': 'Organization ID', 'y': 'Number of Users'}
            )
            
            fig.update_layout(
                showlegend=False,
                height=400,
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Organization statistics
            st.markdown("**Organization Statistics:**")
            st.markdown(f"- **Total Orgs**: {len(org_distribution)}")
            st.markdown(f"- **Largest Org**: {org_distribution.max()} users")
            st.markdown(f"- **Average Size**: {org_distribution.mean():.1f} users")
            st.markdown(f"- **Small Orgs** (<10 users): {len(org_distribution[org_distribution < 10])}")
        else:
            st.info("Organization data not available for visualization")

def display_system_analysis(graph_dfs):
    """Display system-level analysis"""
    
    if 'endpoints' not in graph_dfs:
        return
    
    st.markdown("#### 🏢 System Analysis")
    
    # Calculate entitlements per system
    entitlements_df = graph_dfs['entitlements'].copy()
    entitlements_df['SystemId'] = entitlements_df['id'].astype(str).str.split('_').str[0].astype('Int64')
    
    system_ent_counts = entitlements_df.groupby('SystemId').size().reset_index(name='entitlement_count')
    
    # Merge with endpoint information
    system_analysis = graph_dfs['endpoints'].merge(
        system_ent_counts, 
        left_on='id', 
        right_on='SystemId', 
        how='left'
    ).fillna(0)
    
    # Display top systems
    top_systems = system_analysis.nlargest(10, 'entitlement_count')
    
    if not top_systems.empty:
        fig = px.bar(
            top_systems,
            x='ApplicationCode',
            y='entitlement_count',
            title="Top 10 Systems by Entitlement Count",
            labels={'entitlement_count': 'Number of Entitlements', 'ApplicationCode': 'System'}
        )
        
        fig.update_layout(
            showlegend=False,
            height=400,
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig, use_container_width=True)

def create_data_quality_report(graph_dfs):
    """Create data quality assessment"""
    
    st.markdown("#### ✅ Data Quality Assessment")
    
    quality_checks = []
    
    # Check for missing values in critical fields
    users_df = graph_dfs['users']
    
    # User data quality
    active_users_count = len(users_df[users_df['IsActive'] == True])
    total_users_count = len(users_df)
    
    quality_checks.append({
        'Check': 'Active User Ratio',
        'Result': f"{active_users_count}/{total_users_count} ({active_users_count/total_users_count*100:.1f}%)",
        'Status': '✅ Good' if active_users_count/total_users_count > 0.7 else '⚠️ Review'
    })
    
    # Missing email addresses
    missing_emails = users_df['EmailId'].isna().sum()
    quality_checks.append({
        'Check': 'Email Completeness',
        'Result': f"{total_users_count - missing_emails}/{total_users_count} have emails",
        'Status': '✅ Good' if missing_emails/total_users_count < 0.1 else '⚠️ Review'
    })
    
    # Manager relationships
    has_manager = users_df['ManagerId'].notna().sum()
    quality_checks.append({
        'Check': 'Manager Relationships',
        'Result': f"{has_manager}/{total_users_count} have managers",
        'Status': '✅ Good' if has_manager/total_users_count > 0.8 else '⚠️ Review'
    })
    
    # Access relationships
    users_with_access = len(graph_dfs['entrecon']['UserId'].unique())
    quality_checks.append({
        'Check': 'Users with Access',
        'Result': f"{users_with_access}/{active_users_count} active users",
        'Status': '✅ Good' if users_with_access/active_users_count > 0.9 else '⚠️ Review'
    })
    
    # Display quality report
    quality_df = pd.DataFrame(quality_checks)
    st.dataframe(quality_df, use_container_width=True)
    
    # Overall assessment
    good_checks = len([check for check in quality_checks if check['Status'].startswith('✅')])
    overall_score = good_checks / len(quality_checks) * 100
    
    if overall_score >= 80:
        st.success(f"🎉 **Overall Data Quality: {overall_score:.0f}%** - Excellent for ML training")
    elif overall_score >= 60:
        st.warning(f"⚠️ **Overall Data Quality: {overall_score:.0f}%** - Good, minor improvements possible")
    else:
        st.error(f"❌ **Overall Data Quality: {overall_score:.0f}%** - Requires data cleanup")