import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import distance_matrix, ConvexHull
import folium
from folium.features import DivIcon
from streamlit_folium import st_folium
import requests
import json
import math
import base64

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Planeador Logístico Pro", layout="wide", page_icon="🚚")

# --- CONSTANTES ---
INPUT_FILE = 'instituciones.csv'

# --- FUNCIONES DE UTILIDAD ---
def dms_to_decimal(dms_str):
    if not isinstance(dms_str, str): return None
    clean_str = dms_str.replace('""', '"').strip()
    regex = r"(\d{1,3})[º°](\d{1,2})'([\d.]+).*?([NSWE])"
    import re
    match = re.search(regex, clean_str)
    if match:
        try:
            d, m, s, direction = match.groups()
            dec = float(d) + float(m)/60 + float(s)/3600
            if direction in ['S', 'W']: dec = -dec
            return dec
        except: return None
    return None

def create_buffer_polygon(points, buffer_deg=0.02):
    """Crea un polígono artificial para zonas con < 3 puntos"""
    if len(points) == 1:
        lat, lon = points[0]
        return [
            [lat + buffer_deg, lon + buffer_deg],
            [lat - buffer_deg, lon + buffer_deg],
            [lat - buffer_deg, lon - buffer_deg],
            [lat + buffer_deg, lon - buffer_deg],
            [lat + buffer_deg, lon + buffer_deg]
        ]
    elif len(points) == 2:
        p1, p2 = points
        dx = p2[1] - p1[1]
        dy = p2[0] - p1[0]
        length = math.sqrt(dx*dx + dy*dy)
        if length == 0: return create_buffer_polygon([points[0]])
        nx = -dy / length * (buffer_deg/2)
        ny = dx / length * (buffer_deg/2)
        return [
            [p1[0] + ny, p1[1] + nx],
            [p1[0] - ny, p1[1] - nx],
            [p2[0] - ny, p2[1] - nx],
            [p2[0] + ny, p2[1] + nx],
            [p1[0] + ny, p1[1] + nx]
        ]
    return []

# --- ALGORITMOS DE RUTEO ---
def solve_tsp_round_trip(points, start_idx=0):
    if len(points) < 2: return [0]
    dist_mat = distance_matrix(points, points)
    n = len(points)
    curr = start_idx
    path = [curr]
    visited = {curr}
    while len(visited) < n:
        dists = dist_mat[curr].copy()
        dists[list(visited)] = np.inf
        next_node = np.argmin(dists)
        path.append(next_node)
        visited.add(next_node)
        curr = next_node
    return path

@st.cache_data
def get_osrm_route(coordinates):
    if len(coordinates) < 2: return None, 0, 0, []
    formatted_coords = ";".join([f"{lon},{lat}" for lat, lon in coordinates])
    url = f"https://router.project-osrm.org/route/v1/driving/{formatted_coords}?overview=full&geometries=geojson&steps=true"
    try:
        r = requests.get(url)
        res = r.json()
        if res['code'] == 'Ok':
            route = res['routes'][0]
            legs = []
            for leg in route['legs']:
                legs.append({
                    'dist': leg['distance'], # metros
                    'dur': leg['duration']   # segundos
                })
            return route['geometry'], route['distance']/1000, route['duration']/60, legs
    except: pass
    return None, 0, 0, []

# --- GENERADOR DE REPORTE HTML (INDIVIDUAL) ---
def generate_html_report(zone_name, stats, legs, schools_df, params, depot_coords):
    # Construir filas de itinerario con Links a Google Maps
    itinerary_rows = ""
    route_points = [depot_coords] + schools_df[['lat', 'lon']].values.tolist() + [depot_coords]
    route_names = ["DEPÓSITO CENTRAL"] + schools_df['Nombre Institucion'].tolist() + ["DEPÓSITO CENTRAL"]
    
    for i, leg in enumerate(legs):
        origin = route_names[i]
        dest = route_names[i+1]
        dist = f"{leg['dist']/1000:.1f} km"
        time = f"{int(leg['dur']//60)} min"
        
        lat_a, lon_a = route_points[i]
        lat_b, lon_b = route_points[i+1]
        gmaps_link = f"https://www.google.com/maps/dir/?api=1&origin={lat_a},{lon_a}&destination={lat_b},{lon_b}&travelmode=driving"
        
        itinerary_rows += f"""
        <tr>
            <td style="text-align:center;">{i+1}</td>
            <td>{origin}</td>
            <td>{dest}</td>
            <td style="text-align:right;">{dist}</td>
            <td style="text-align:right;">{time}</td>
            <td style="text-align:center;"><a href="{gmaps_link}" target="_blank" style="text-decoration:none; color:#1a73e8; font-weight:bold;">📍 Ver Mapa</a></td>
        </tr>
        """

    cargo_rows = ""
    cols_grados = ['1º Grado', '2º Grado', '3º Grado', '4º Grado', '5º Grado', '6º Grado']
    
    for _, row in schools_df.iterrows():
        grados_td = "".join([f'<td style="text-align:center;">{row[c]}</td>' for c in cols_grados])
        cargo_rows += f"""
        <tr>
            <td style="text-align:center;"><b>{row['Orden']}</b></td>
            <td>{row['Nombre Institucion']}</td>
            <td>{row['Distrito']}</td>
            {grados_td}
            <td style="text-align:center; font-weight:bold;">{row['Total_Alumnos']}</td>
        </tr>
        """

    html = f"""
    <html>
    <head>
        <title>Reporte Logístico - Zona {zone_name}</title>
        <style>
            body {{ font-family: 'Helvetica', 'Arial', sans-serif; padding: 40px; color: #333; }}
            .header {{ display: flex; justify-content: space-between; border-bottom: 3px solid #333; padding-bottom: 20px; margin-bottom: 30px; }}
            .title {{ font-size: 28px; font-weight: bold; text-transform: uppercase; }}
            .meta {{ text-align: right; font-size: 14px; line-height: 1.5; }}
            h2 {{ border-bottom: 1px solid #ccc; padding-bottom: 10px; margin-top: 40px; color: #444; font-size: 18px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 12px; }}
            th {{ background-color: #f2f2f2; border: 1px solid #ccc; padding: 8px; text-align: left; }}
            td {{ border: 1px solid #ccc; padding: 8px; }}
            .box {{ background: #f9f9f9; padding: 15px; border-radius: 5px; border: 1px solid #eee; margin-bottom: 20px; }}
            .cost-val {{ font-weight: bold; color: #2ecc71; font-size: 1.2em; }}
            @media print {{ .no-print {{ display: none; }} a {{ text-decoration: none; color: #000; }} }}
        </style>
    </head>
    <body>
        <div class="no-print" style="margin-bottom:20px; padding:15px; background:#e8f0fe; border:1px solid #b3d4fc; text-align:center;">
            <button onclick="window.print()" style="font-size:16px; padding:10px 20px; cursor:pointer; background:#1a73e8; color:white; border:none; border-radius:4px;">🖨️ Imprimir Reporte / Guardar PDF</button>
        </div>
        <div class="header">
            <div>
                <div class="title">Hoja de Ruta Logística</div>
                <div style="font-size: 20px; font-weight:bold; margin-top:10px;">ZONA {zone_name}</div>
            </div>
            <div class="meta">
                <div><b>Distancia Total:</b> {stats['dist']:.1f} km</div>
                <div><b>Tiempo Estimado:</b> {int(stats['time_drive']//60)}h {int(stats['time_drive']%60)}m</div>
                <div><b>Total Escuelas:</b> {len(schools_df)}</div>
                <div><b>Carga Total:</b> {schools_df['Total_Alumnos'].sum()} kits</div>
            </div>
        </div>
        <div class="box">
            <div style="display:flex; justify-content:space-between;">
                <div>
                    <strong>Parámetros de Costo:</strong><br>
                    Combustible: {params['fuel_price']:,} Gs/km<br>
                    Operador: {params['labor_price_student']:,} Gs/Alumno<br>
                    Tiempo Parada: {params['stop_time']} min
                </div>
                <div style="text-align:right;">
                    <strong>Estimación Económica:</strong><br>
                    Combustible: {stats['cost_fuel']:,.0f} Gs<br>
                    Operativo: {stats['cost_hr']:,.0f} Gs<br>
                    <div style="margin-top:5px; border-top:1px solid #ccc; padding-top:5px;">
                        Total: <span class="cost-val">{stats['cost_total']:,.0f} Gs</span>
                    </div>
                </div>
            </div>
        </div>
        <h2>1. Itinerario de Viaje</h2>
        <table>
            <thead>
                <tr>
                    <th width="5%">#</th>
                    <th width="30%">Origen</th>
                    <th width="30%">Destino</th>
                    <th width="10%" style="text-align:right;">Distancia</th>
                    <th width="10%" style="text-align:right;">Tiempo</th>
                    <th width="15%" style="text-align:center;">Navegación</th>
                </tr>
            </thead>
            <tbody>{itinerary_rows}</tbody>
        </table>
        <h2>2. Detalle de Carga</h2>
        <table>
            <thead>
                <tr>
                    <th width="5%">Ord</th>
                    <th width="30%">Institución</th>
                    <th width="15%">Distrito</th>
                    <th width="7%" style="text-align:center;">1º</th>
                    <th width="7%" style="text-align:center;">2º</th>
                    <th width="7%" style="text-align:center;">3º</th>
                    <th width="7%" style="text-align:center;">4º</th>
                    <th width="7%" style="text-align:center;">5º</th>
                    <th width="7%" style="text-align:center;">6º</th>
                    <th width="8%" style="text-align:center;">Total</th>
                </tr>
            </thead>
            <tbody>{cargo_rows}</tbody>
        </table>
        <div style="margin-top: 50px; display:flex; justify-content:space-between; page-break-inside:avoid;">
            <div style="border-top:1px solid #000; width:40%; padding-top:10px; font-size:12px;">Firma Coordinador Zonal</div>
            <div style="border-top:1px solid #000; width:40%; padding-top:10px; font-size:12px;">Firma Chofer Responsable</div>
        </div>
    </body>
    </html>
    """
    return html

# --- GESTIÓN DE ESTADO ---
if 'data' not in st.session_state: st.session_state.data = None
if 'zones_created' not in st.session_state: st.session_state.zones_created = False
if 'zone_depots' not in st.session_state: st.session_state.zone_depots = {}
if 'picking_depot' not in st.session_state: st.session_state.picking_depot = False
# Nuevo: Almacén para el reporte maestro { 'Zona 1': {datos...}, 'Zona 2': {datos...} }
if 'master_report' not in st.session_state: st.session_state.master_report = {}

# --- CARGA DE DATOS ---
def load_data():
    try: df = pd.read_csv(INPUT_FILE, encoding='utf-8')
    except: df = pd.read_csv(INPUT_FILE, encoding='latin-1')
    df.columns = df.columns.str.strip()
    df['lat'] = df['latitud aproximada'].astype(str).apply(dms_to_decimal)
    df['lon'] = df['longitud aproximada'].astype(str).apply(dms_to_decimal)
    cols_grados = ['1º Grado', '2º Grado', '3º Grado', '4º Grado', '5º Grado', '6º Grado']
    for c in cols_grados:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(int)
    df['Total_Alumnos'] = df[cols_grados].sum(axis=1) if all(c in df.columns for c in cols_grados) else 0
    df = df.dropna(subset=['lat', 'lon']).copy()
    df['ID_Interno'] = range(len(df))
    return df

# --- INTERFAZ PRINCIPAL ---
st.title("🚚 Sistema de Planeación Logística Escolar")

# 1. SIDEBAR: CONFIGURACIÓN
with st.sidebar:
    st.header("1. Configuración General")
    
    with st.expander("💰 Parámetros de Costos", expanded=True):
        costo_km = st.number_input("Costo Combustible (Gs/Km)", 5000, 20000, 8500, step=500)
        costo_alumno = st.number_input("Costo Operador (Gs/Alumno)", 500, 50000, 5000, step=500)
        tiempo_parada = st.number_input("Tiempo por Parada (min)", 5, 120, 30, step=5)
    
    st.divider()
    
    st.subheader("⚙️ Generación de Zonas")
    n_zones = st.slider("Zonas Macro", 5, 20, 10)
    limit_type = st.radio("Criterio de Límite", ["Por Cantidad de Escuelas", "Por Cantidad de Alumnos"])
    
    if limit_type == "Por Cantidad de Escuelas":
        max_val = st.number_input("Máx Escuelas por Ruta", 20, 100, 45)
        constraint_col = 'Escuelas'
    else:
        max_val = st.number_input("Máx Alumnos por Ruta", 1000, 10000, 5000)
        constraint_col = 'Total_Alumnos'
        
    if st.button("🔄 Generar/Resetear Zonas"):
        df = load_data()
        kmeans = KMeans(n_clusters=n_zones, random_state=42, n_init=10)
        df['cluster'] = kmeans.fit_predict(df[['lat', 'lon']])
        df['zona'] = df['cluster'].astype(str)
        counts = df['cluster'].value_counts()
        for c_id in counts.index:
            mask = df['cluster'] == c_id
            subset = df.loc[mask]
            val = len(subset) if constraint_col == 'Escuelas' else subset['Total_Alumnos'].sum()
            if val > max_val:
                n_sub = int(np.ceil(val / max_val))
                sub_k = KMeans(n_clusters=n_sub, random_state=42, n_init=10)
                sub_labels = sub_k.fit_predict(subset[['lat', 'lon']])
                df.loc[mask, 'zona'] = df.loc[mask, 'cluster'].astype(str) + "-" + pd.Series(sub_labels, index=subset.index).astype(str)
        
        unique_zones = df['zona'].unique()
        st.session_state.zone_depots = {}
        # Reset master report on regeneration
        st.session_state.master_report = {}
        
        for z in unique_zones:
            z_data = df[df['zona'] == z]
            st.session_state.zone_depots[z] = [z_data['lat'].mean(), z_data['lon'].mean()]

        st.session_state.data = df
        st.session_state.zones_created = True
        st.rerun()
    
    # --- REPORTE MAESTRO ---
    st.divider()
    st.subheader("📊 Reporte Maestro de Proyecto")
    if st.session_state.master_report:
        df_master = pd.DataFrame.from_dict(st.session_state.master_report, orient='index')
        df_master.index.name = 'Zona'
        df_master.reset_index(inplace=True)
        
        # Totales
        st.markdown("---")
        c1, c2 = st.columns(2)
        c1.metric("Total Presupuesto", f"{df_master['Costo Total'].sum():,.0f} Gs")
        c2.metric("Total Alumnos", f"{df_master['Total Alumnos'].sum():,.0f}")
        
        with st.expander("Ver Tabla Consolidada"):
            st.dataframe(df_master, hide_index=True)
            
            # Descargar CSV
            csv = df_master.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Descargar Reporte Maestro (CSV)",
                csv,
                "reporte_maestro_logistica.csv",
                "text/csv",
                key='download-master'
            )
    else:
        st.caption("Calcule rutas para alimentar el reporte maestro.")

# --- CUERPO PRINCIPAL ---
if not st.session_state.zones_created:
    st.info("👈 Configura los parámetros en la barra lateral y pulsa 'Generar Zonas' para comenzar.")
else:
    df = st.session_state.data
    
    # 2. EDITOR DE ZONAS
    with st.expander("🛠️ Editor de Zonas (Fusionar/Dividir)", expanded=False):
        c1, c2 = st.columns(2)
        zonas_disp = sorted(df['zona'].unique())
        with c1:
            st.markdown("#### Fusionar Zonas")
            z_a = st.selectbox("Zona A", zonas_disp, key='za')
            z_b = st.selectbox("Zona B", [z for z in zonas_disp if z != z_a], key='zb')
            if st.button("🔗 Fusionar"):
                new_name = f"{z_a}+{z_b.split('-')[-1]}"
                df.loc[df['zona'].isin([z_a, z_b]), 'zona'] = new_name
                z_data = df[df['zona'] == new_name]
                st.session_state.zone_depots[new_name] = [z_data['lat'].mean(), z_data['lon'].mean()]
                if z_a in st.session_state.zone_depots: del st.session_state.zone_depots[z_a]
                if z_b in st.session_state.zone_depots: del st.session_state.zone_depots[z_b]
                # Limpiar reporte maestro de zonas viejas
                if z_a in st.session_state.master_report: del st.session_state.master_report[z_a]
                if z_b in st.session_state.master_report: del st.session_state.master_report[z_b]
                
                st.session_state.data = df
                st.rerun()
        with c2:
            st.markdown("#### Dividir Zona")
            z_split = st.selectbox("Zona a Dividir", zonas_disp, key='zsplit')
            n_parts = st.number_input("Dividir en N partes", 2, 5, 2)
            if st.button("✂️ Dividir"):
                mask = df['zona'] == z_split
                subset = df.loc[mask]
                if len(subset) >= n_parts:
                    sub_k = KMeans(n_clusters=n_parts, random_state=42, n_init=10)
                    sub_labs = sub_k.fit_predict(subset[['lat', 'lon']])
                    new_labels = z_split + "." + pd.Series(sub_labs, index=subset.index).astype(str)
                    df.loc[mask, 'zona'] = new_labels
                    if z_split in st.session_state.zone_depots: del st.session_state.zone_depots[z_split]
                    if z_split in st.session_state.master_report: del st.session_state.master_report[z_split]
                    
                    for sub_z in new_labels.unique():
                        sub_data = df[df['zona'] == sub_z]
                        st.session_state.zone_depots[sub_z] = [sub_data['lat'].mean(), sub_data['lon'].mean()]
                    st.session_state.data = df
                    st.rerun()

    # 3. VISUALIZACIÓN Y RUTAS
    col_map, col_details = st.columns([2, 1])
    
    with col_details:
        st.subheader("📊 Gestión de Zona")
        selected_zone = st.selectbox("Seleccionar Zona de Trabajo", sorted(df['zona'].unique()))
        
        # --- GESTIÓN DEPÓSITO ---
        st.markdown("---")
        c_dep_title, c_dep_btn = st.columns([2,1])
        with c_dep_title: st.markdown("##### 📍 Punto de Partida")
        
        if st.session_state.picking_depot:
            st.warning("👇 Haz clic en el mapa para fijar el depósito")
            if st.button("Cancelar"):
                st.session_state.picking_depot = False
                st.rerun()
        else:
            if c_dep_btn.button("Definir en Mapa"):
                st.session_state.picking_depot = True
                st.rerun()

        if selected_zone not in st.session_state.zone_depots:
             z_d = df[df['zona'] == selected_zone]
             st.session_state.zone_depots[selected_zone] = [z_d['lat'].mean(), z_d['lon'].mean()]
        
        current_depot = st.session_state.zone_depots[selected_zone]
        st.caption(f"Lat: {current_depot[0]:.5f}, Lon: {current_depot[1]:.5f}")
        st.markdown("---")

        # Lógica de Ruteo
        zone_df = df[df['zona'] == selected_zone].copy()
        points = [[current_depot[0], current_depot[1]]] + zone_df[['lat', 'lon']].values.tolist()
        tsp_indices = solve_tsp_round_trip(np.array(points), start_idx=0)
        ordered_schools_idx = [i-1 for i in tsp_indices if i != 0]
        zone_df_ordered = zone_df.iloc[ordered_schools_idx].copy()
        zone_df_ordered['Orden'] = range(1, len(zone_df_ordered)+1)
        
        st.info(f"**Escuelas:** {len(zone_df_ordered)} | **Alumnos:** {zone_df_ordered['Total_Alumnos'].sum()}")
        
        if st.button("🛣️ Calcular Ruta Vial (OSRM) + Reporte", type="primary"):
            with st.spinner("Optimizando ruta y calculando costos..."):
                final_coords = [[current_depot[0], current_depot[1]]] + zone_df_ordered[['lat', 'lon']].values.tolist() + [[current_depot[0], current_depot[1]]]
                geo_json, dist_km, dur_min, legs = get_osrm_route(final_coords)
                
                if geo_json:
                    # CÁLCULO DE COSTOS
                    costo_combustible_total = dist_km * costo_km
                    total_carga_alumnos = zone_df_ordered['Total_Alumnos'].sum()
                    costo_personal_total = total_carga_alumnos * costo_alumno
                    costo_total = costo_combustible_total + costo_personal_total
                    
                    st.session_state['last_route_geo'] = geo_json
                    st.session_state['last_route_zone'] = selected_zone
                    st.session_state['last_route_legs'] = legs
                    st.session_state['last_route_stats'] = {
                        'dist': dist_km, 'time_drive': dur_min, 
                        'cost_total': costo_total, 'cost_fuel': costo_combustible_total, 'cost_hr': costo_personal_total
                    }
                    st.session_state['last_route_params'] = {
                        'fuel_price': costo_km, 'labor_price_student': costo_alumno, 'stop_time': tiempo_parada
                    }
                    
                    # --- ACTUALIZAR REPORTE MAESTRO ---
                    report_entry = {
                        'Cantidad Escuelas': len(zone_df_ordered),
                        'Total Alumnos': total_carga_alumnos,
                        'Km Trayecto': round(dist_km, 2),
                        'Costo Combustible': round(costo_combustible_total, 0),
                        'Costo Operador': round(costo_personal_total, 0),
                        'Costo Total': round(costo_total, 0)
                    }
                    # Agregar desglose por grados
                    grades_sum = zone_df_ordered[['1º Grado', '2º Grado', '3º Grado', '4º Grado', '5º Grado', '6º Grado']].sum().to_dict()
                    report_entry.update(grades_sum)
                    
                    st.session_state.master_report[selected_zone] = report_entry
                    st.success("✅ Ruta calculada y agregada al Reporte Maestro")
                    
                else:
                    st.error("Error API OSRM.")

        # MOSTRAR RESULTADOS SI EXISTEN
        if 'last_route_stats' in st.session_state and st.session_state.get('last_route_zone') == selected_zone:
            stats = st.session_state['last_route_stats']
            legs = st.session_state.get('last_route_legs', [])
            params = st.session_state.get('last_route_params', {'fuel_price':0, 'labor_price_student':0, 'stop_time':0})
            
            st.markdown("### 💰 Estimación Económica")
            c1, c2, c3 = st.columns(3)
            c1.metric("Total (Gs)", f"{stats['cost_total']:,.0f}")
            c2.metric("Combustible", f"{stats['cost_fuel']:,.0f}")
            c3.metric("Operativo (x Alumno)", f"{stats['cost_hr']:,.0f}")
            
            st.markdown(f"**Recorrido:** {stats['dist']:.1f} km | **Tiempo Total:** {int(stats['time_drive']//60)}h {int(stats['time_drive']%60)}m (+ paradas)")
            
            # --- GENERACIÓN DE REPORTE ---
            report_html = generate_html_report(selected_zone, stats, legs, zone_df_ordered, params, current_depot)
            
            # Botón para abrir reporte
            import streamlit.components.v1 as components
            with st.expander("📄 Ver Reporte Oficial de Logística", expanded=True):
                st.components.v1.html(report_html, height=600, scrolling=True)
                
                b64 = base64.b64encode(report_html.encode()).decode()
                href = f'<a href="data:text/html;base64,{b64}" download="Reporte_Zona_{selected_zone}.html" style="text-decoration:none;"><button style="background-color:#4CAF50;color:white;padding:8px 12px;border:none;border-radius:4px;cursor:pointer;">📥 Descargar HTML del Reporte</button></a>'
                st.markdown(href, unsafe_allow_html=True)

        with st.expander("Ver lista de carga completa"):
            cols_show = ['Orden', 'Nombre Institucion', 'Distrito'] + ['1º Grado', '2º Grado', '3º Grado', '4º Grado', '5º Grado', '6º Grado'] + ['Total_Alumnos']
            st.dataframe(zone_df_ordered[cols_show], hide_index=True)

    with col_map:
        # MAPA
        m = folium.Map(location=[df['lat'].mean(), df['lon'].mean()], zoom_start=7)
        
        def get_dark_color(idx):
            hue = (idx * 137.508) % 360
            return f"hsl({hue}, 100%, 30%)" # Saturación Max, Luz Baja
        
        unique_zones = sorted(df['zona'].unique())
        
        for i, zona in enumerate(unique_zones):
            z_df = df[df['zona'] == zona]
            color = get_dark_color(i)
            
            try:
                pts = z_df[['lat', 'lon']].values
                hull_pts = []
                if len(z_df) < 3:
                    hull_pts = create_buffer_polygon(pts)
                else:
                    hull = ConvexHull(pts)
                    hull_pts = pts[hull.vertices].tolist()
                
                folium.Polygon(
                    locations=hull_pts, color=color, weight=2,
                    fill=True, fill_color=color, fill_opacity=0.2, # Transparencia ajustada
                    tooltip=f"Zona {zona}"
                ).add_to(m)
                
                center_lat = np.mean(pts[:,0])
                center_lon = np.mean(pts[:,1])
                folium.map.Marker(
                    [center_lat, center_lon],
                    icon=DivIcon(
                        icon_size=(150,36), icon_anchor=(75,18),
                        html=f'<div style="font-size: 10pt; font-weight: 900; color: {color}; text-shadow: -1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff, 1px 1px 0 #fff;">Zona {zona}</div>',
                    )
                ).add_to(m)
            except: pass

            if zona == selected_zone and 'last_route_zone' in st.session_state and st.session_state['last_route_zone'] == selected_zone:
                for _, row in zone_df_ordered.iterrows():
                    folium.map.Marker(
                        [row['lat'], row['lon']],
                        icon=DivIcon(
                            icon_size=(24,24), icon_anchor=(12,12),
                            html=f'<div style="background:{color}; color:white; border-radius:50%; text-align:center; line-height:24px; font-weight:bold; border:2px solid white; box-shadow:0 0 3px black;">{row["Orden"]}</div>'
                        ),
                        tooltip=f"{row['Orden']}. {row['Nombre Institucion']}"
                    ).add_to(m)
            else:
                for _, row in z_df.iterrows():
                    folium.CircleMarker(
                        location=[row['lat'], row['lon']],
                        radius=4, color=color, fill=True, fill_opacity=0.9, weight=1,
                        tooltip=f"Zona {zona}: {row['Nombre Institucion']}"
                    ).add_to(m)

        if selected_zone in st.session_state.zone_depots:
            d_lat, d_lon = st.session_state.zone_depots[selected_zone]
            folium.Marker(
                [d_lat, d_lon],
                tooltip=f"Depósito {selected_zone}",
                icon=folium.Icon(color='black', icon='home', prefix='fa'),
                draggable=False
            ).add_to(m)

        if 'last_route_geo' in st.session_state and st.session_state.get('last_route_zone') == selected_zone:
            folium.GeoJson(
                st.session_state['last_route_geo'],
                style_function=lambda x: {'color': 'black', 'weight': 4, 'opacity': 0.7, 'dashArray': '5, 5'}
            ).add_to(m)

        map_data = st_folium(m, width=None, height=700)
        
        if st.session_state.picking_depot and map_data['last_clicked']:
            clicked = map_data['last_clicked']
            st.session_state.zone_depots[selected_zone] = [clicked['lat'], clicked['lng']]
            st.session_state.picking_depot = False
            st.rerun()