## ADDED Requirements

### Requirement: Predicción interactiva de un turno individual
La aplicación SHALL permitir al usuario ingresar los atributos de un turno (edad, género, barrio, fechas/lead time, envío de SMS, comorbilidades, beca y clima del día) y mostrar la probabilidad estimada de no-show.

#### Scenario: Estimación de un turno
- **WHEN** el usuario completa los atributos de un turno y confirma
- **THEN** la app SHALL mostrar la probabilidad de no-show del turno usando el modelo persistido

#### Scenario: Reutilización del pipeline de entrenamiento
- **WHEN** la app transforma los atributos ingresados
- **THEN** SHALL usar el mismo pipeline de preprocesado persistido con el modelo, sin recodificar manualmente

### Requirement: Recomendación de acción según riesgo
La aplicación SHALL traducir la probabilidad de no-show en una acción de negocio recomendada según bandas de riesgo (por ejemplo bajo: sin acción; medio: recordatorio SMS; alto: llamado + sobreturno controlado).

#### Scenario: Turno de alto riesgo
- **WHEN** la probabilidad de no-show supera el umbral alto configurado
- **THEN** la app SHALL recomendar la acción de mayor intensidad (llamado y/o sobreturno controlado)

#### Scenario: Turno de bajo riesgo
- **WHEN** la probabilidad de no-show está por debajo del umbral bajo
- **THEN** la app SHALL indicar que no se requiere acción preventiva

### Requirement: Modo lote sobre la agenda del día
La aplicación SHALL permitir cargar un conjunto de turnos (por ejemplo un CSV con la agenda del día) y devolver una tabla rankeada por riesgo de no-show con la acción sugerida por turno.

#### Scenario: Scoring de una agenda
- **WHEN** el usuario sube un archivo con múltiples turnos válidos
- **THEN** la app SHALL mostrar cada turno con su probabilidad de no-show y su acción recomendada, ordenados de mayor a menor riesgo

#### Scenario: Estimación de valor de negocio
- **WHEN** se procesa una agenda del día
- **THEN** la app SHALL estimar una métrica de valor (por ejemplo horas-profesional potencialmente recuperadas o sobreturnos sugeridos) coherente con la propuesta de valor del proyecto
