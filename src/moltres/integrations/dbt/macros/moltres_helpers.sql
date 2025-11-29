{# Moltres helper macros for dbt #}

{# Get Moltres connection string from dbt profile #}
{% macro moltres_connection_string() %}
  {% set profile = target %}
  {% set dsn = "postgresql://" %}
  {% if profile.user %}
    {% set dsn = dsn ~ profile.user %}
    {% if profile.password %}
      {% set dsn = dsn ~ ":" ~ profile.password %}
    {% endif %}
    {% set dsn = dsn ~ "@" %}
  {% endif %}
  {% if profile.host %}
    {% set dsn = dsn ~ profile.host %}
    {% if profile.port %}
      {% set dsn = dsn ~ ":" ~ profile.port %}
    {% endif %}
  {% endif %}
  {% if profile.dbname %}
    {% set dsn = dsn ~ "/" ~ profile.dbname %}
  {% endif %}
  {{ return(dsn) }}
{% endmacro %}

