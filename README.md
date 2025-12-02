# **🚚 Sistema de Planeación Logística Escolar (Pro)**

Este es un **ERP Logístico liviano y potente** diseñado para optimizar la distribución de materiales escolares a nivel nacional. Utiliza inteligencia artificial (K-Means) para agrupar escuelas, algoritmos de grafos (TSP) para ordenar las rutas y APIs de geolocalización (OSRM) para calcular distancias reales en carretera.

## **🌟 Características Principales**

* **Zonificación Inteligente:** Agrupa automáticamente 500+ escuelas en zonas logísticas equilibradas, ya sea por cantidad de paradas o por volumen de carga (alumnos).  
* **Ruteo Óptimo (TSP & OSRM):** Calcula el orden de visita más eficiente (Vecino más cercano) y traza la ruta real sobre el mapa vial.  
* **Gestión Financiera:** Estima costos operativos en tiempo real (Combustible \+ Mano de Obra por Alumno \+ Tiempo en Parada).  
* **Reportes Ejecutivos:** Genera hojas de ruta listas para imprimir con mapas, itinerarios paso a paso y tablas de carga segregadas por grado (1º a 6º).  
* **Editor Visual:** Permite fusionar o dividir zonas y ajustar la ubicación de los depósitos de partida directamente en el mapa.

## **🚀 Instalación y Ejecución Local**

Si deseas correr este proyecto en tu propia máquina:

1. **Clonar el repositorio:**  
   git clone \<tu-link-del-repo\>  
   cd \<nombre-del-repo\>

2. Instalar dependencias:  
   Asegúrate de tener Python instalado. Luego ejecuta:  
   pip install \-r requirements.txt

3. **Ejecutar la aplicación:**  
   streamlit run app\_logistica.py

4. Abrir en el navegador:  
   La terminal te mostrará una dirección (usualmente http://localhost:8501).

## **☁️ Despliegue en Streamlit Community Cloud**

Este proyecto está listo para la nube. Pasos:

1. Sube los archivos app\_logistica.py, instituciones.csv y requirements.txt a un repositorio de GitHub.  
2. Ve a [share.streamlit.io](https://share.streamlit.io/).  
3. Conecta tu cuenta de GitHub y selecciona este repositorio.  
4. ¡Listo\! Tu ERP estará accesible desde cualquier lugar.

## **🛠️ Guía de Uso Rápida**

1. **Configuración (Barra Lateral):** Define el costo del combustible, el costo por alumno del operador y el criterio de agrupación (ej. Máx 5000 alumnos por camión). Pulsa "Generar Zonas".  
2. **Gestión de Zonas:** Usa el selector "Editor de Zonas" si necesitas unir dos zonas pequeñas o dividir una muy grande manualmente.  
3. **Cálculo de Ruta:**  
   * Selecciona una zona en el tablero.  
   * Verifica o mueve el punto de partida (Depósito) en el mapa.  
   * Presiona **"Calcular Ruta Vial (OSRM)"**.  
4. **Impresión:** Una vez calculada la ruta, presiona **"Ver Reporte Oficial"** y luego el botón de imprimir dentro del reporte para generar el PDF entregable al chofer.

## **📁 Estructura de Archivos**

* app\_logistica.py: Código fuente principal de la aplicación.  
* instituciones.csv: Base de datos con geolocalización y matrícula por grado.  
* requirements.txt: Lista de librerías necesarias.

*Desarrollado para optimización logística escolar en Paraguay.*