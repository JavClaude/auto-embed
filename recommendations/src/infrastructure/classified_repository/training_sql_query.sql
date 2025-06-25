SELECT
    reference as classified_ref,
    vehicle_crit_air,
    vehicle_seats,
    vehicle_mileage,
    vehicle_year,
    vehicle_doors,
    vehicle_trunk_volume,
    vehicle_refined_quotation,
    vehicle_power_din,
    vehicle_rated_horse_power,
    vehicle_max_power,
    vehicle_consumption,
    vehicle_co2,
    constructor_warranty_duration,
    price,
    vehicle_weight,
    vehicle_cubic,
    vehicle_length,
    vehicle_height,
    vehicle_width,
    initial_price,
    vehicle_price_new,
    customer_type,
    substr(zip_code, 1, 2) AS zip_code,
    vehicle_category,
    vehicle_make,
    vehicle_model,
    vehicle_version,
    vehicle_gearbox,
    vehicle_energy,
    vehicle_origin,
    vehicle_external_color,
    vehicle_internal_color,
    vehicle_four_wheel_drive,
    vehicle_pollution_norm,
    vehicle_condition,
    vehicle_motorization,
    vehicle_commercial_name
FROM data_datalakehouse_prod.dim_annonce_lc 
-- annonces dépubliées entre le 22 mai et 22 juin
WHERE (CONCAT(year, '-', month, '-', day) BETWEEN '2025-03-22' AND '2025-06-22'
    AND outdated_state = 'outdated'
    AND publication_end_date >= cast('2025-03-22' as date)
    AND f_status_refused = false 
    AND f_status_scam = false)
-- annonces en ligne au 22
OR (
    CONCAT(year, '-', month, '-', day) = '2025-06-22'
    AND outdated_state = 'current'
    AND f_status_online = true
)