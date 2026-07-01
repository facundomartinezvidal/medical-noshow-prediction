## ADDED Requirements

### Requirement: Ingesta y validación del dataset de turnos
El sistema SHALL cargar el dataset de turnos desde un CSV y validar que su esquema coincida con las 14 columnas esperadas (`PatientId, AppointmentID, Gender, ScheduledDay, AppointmentDay, Age, Neighbourhood, Scholarship, Hipertension, Diabetes, Alcoholism, Handcap, SMS_received, No-show`) antes de continuar el pipeline.

#### Scenario: CSV con esquema válido
- **WHEN** se carga `no-show-dataset.csv` con las 14 columnas esperadas
- **THEN** el sistema retorna un DataFrame con 110.527 filas y tipos parseados (fechas como datetime, target como binaria)

#### Scenario: CSV con columnas faltantes o renombradas
- **WHEN** el CSV no contiene alguna columna esperada
- **THEN** el sistema SHALL abortar con un error explícito que nombra las columnas faltantes, sin producir un dataset parcial

### Requirement: Normalización de la variable objetivo
El sistema SHALL transformar la columna `No-show` (valores "Yes"/"No") en una variable binaria `no_show` donde 1 significa que el paciente faltó y 0 que asistió.

#### Scenario: Mapeo de la etiqueta
- **WHEN** una fila tiene `No-show == "Yes"`
- **THEN** su valor de `no_show` SHALL ser 1; y cuando `No-show == "No"`, SHALL ser 0

### Requirement: Limpieza de datos inválidos
El sistema SHALL detectar y corregir registros con valores imposibles antes del modelado, en particular edades negativas y lead times negativos.

#### Scenario: Edad negativa
- **WHEN** una fila tiene `Age` menor a 0
- **THEN** el sistema SHALL descartar o corregir ese registro y dejar constancia del conteo de registros afectados

#### Scenario: Fecha de turno anterior al agendamiento
- **WHEN** `AppointmentDay` es anterior a la fecha de `ScheduledDay` (lead time negativo)
- **THEN** el sistema SHALL tratar el registro como dato inválido (descartar o clip a 0) y registrar el conteo

### Requirement: Ingeniería de características
El sistema SHALL derivar las características predictoras a partir de las columnas crudas, incluyendo como mínimo `lead_time_days`, el día de la semana y el mes del turno, indicadores de comorbilidad y una versión agrupada de barrios de baja frecuencia.

#### Scenario: Cálculo del lead time
- **WHEN** se procesa un turno con `ScheduledDay` y `AppointmentDay`
- **THEN** `lead_time_days` SHALL ser la diferencia en días entre la fecha del turno y la del agendamiento

#### Scenario: Dataset procesado reproducible
- **WHEN** se ejecuta el pipeline de preparación sobre el dataset crudo
- **THEN** el sistema SHALL producir un dataset procesado determinístico, reutilizable tanto por el entrenamiento como por la aplicación
