-- Optional bootstrap/verification SQL for the event-style assignment model.
-- The target database already has these objects in "ew-kc-schema"; keep this
-- file with deployments so the expected contract is explicit.

CREATE TABLE IF NOT EXISTS "ew-kc-schema".t_group_assignment (
    id BIGINT GENERATED ALWAYS AS IDENTITY,
    employee_id BIGINT,
    group_id BIGINT,
    "datetime" TIMESTAMP(6) WITHOUT TIME ZONE,
    comment VARCHAR(1000),
    status VARCHAR
);

CREATE INDEX IF NOT EXISTS ix_group_assignment_group_latest
    ON "ew-kc-schema".t_group_assignment (group_id, "datetime" DESC, id DESC);

CREATE INDEX IF NOT EXISTS ix_group_assignment_employee_latest
    ON "ew-kc-schema".t_group_assignment (employee_id, "datetime" DESC, id DESC);

CREATE OR REPLACE VIEW "ew-kc-schema".v_group_assignment AS
WITH latest_assignment AS (
    SELECT DISTINCT ON (tga.group_id)
        tga.id,
        tga.employee_id,
        tga.group_id,
        tga."datetime",
        tga.comment,
        tga.status
    FROM "ew-kc-schema".t_group_assignment tga
    ORDER BY tga.group_id, tga."datetime" DESC NULLS LAST, tga.id DESC
)
SELECT
    la.id,
    la.employee_id,
    me.employee_name,
    la.group_id,
    la."datetime",
    la.comment,
    la.status
FROM latest_assignment la
LEFT JOIN "ew-kc-schema".m_employee me
    ON me.employee_id = la.employee_id;
