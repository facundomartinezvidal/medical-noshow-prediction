# weather-integration Specification

## Purpose
TBD - created by archiving change noshow-prediction. Update Purpose after archive.
## Requirements
### Requirement: Selección de la estación meteorológica de Vitória
El sistema SHALL filtrar el dataset horario de INMET 2016 a la estación `A612` (Vitória, Espírito Santo), que corresponde a la ciudad de origen del dataset de turnos.

#### Scenario: Filtrado por estación
- **WHEN** se procesa `weather-dataset.csv` (529 estaciones)
- **THEN** el sistema SHALL retener únicamente las filas con `ESTACAO == "A612"`

#### Scenario: Manejo del archivo grande
- **WHEN** el archivo de clima supera cientos de MB
- **THEN** el sistema SHALL procesarlo sin cargar el archivo completo en memoria de una sola vez (lectura por chunks/streaming) y cachear el resultado diario

### Requirement: Agregación horaria a diaria
El sistema SHALL agregar los registros horarios de la estación a granularidad diaria, produciendo como mínimo precipitación total del día, temperatura máxima, mínima y media, y un indicador `is_rainy`.

#### Scenario: Agregación de un día
- **WHEN** existen 24 registros horarios para una fecha
- **THEN** el sistema SHALL producir una única fila diaria con la suma de precipitación y los estadísticos de temperatura, con nombres de columna normalizados desde el portugués original

#### Scenario: Indicador de lluvia
- **WHEN** la precipitación total diaria es mayor a 0
- **THEN** `is_rainy` SHALL ser 1; en caso contrario SHALL ser 0

### Requirement: Cruce del clima con los turnos por fecha
El sistema SHALL unir las características climáticas diarias al dataset de turnos usando la fecha de `AppointmentDay` como clave.

#### Scenario: Join por fecha del turno
- **WHEN** un turno tiene una `AppointmentDay` dentro del rango cubierto por el clima (2016-04-29 a 2016-06-08)
- **THEN** el turno SHALL quedar enriquecido con las variables de clima de ese día

#### Scenario: Fecha sin clima disponible
- **WHEN** un turno cae en una fecha sin registro de clima
- **THEN** el sistema SHALL manejar el faltante de forma explícita (imputación o marca) sin romper el pipeline

