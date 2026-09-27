# /// script
# dependencies = ["jinja2"]
# ///
import json
from jinja2.sandbox import ImmutableSandboxedEnvironment
from jinja2.ext import loopcontrols
env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
env.filters["tojson"] = lambda x, indent=None: json.dumps(x, ensure_ascii=False, indent=indent)
cases = [
  ("whitespace", "a\n  {% if true %}\n    b\n  {% endif %}\nc {{- ' d ' -}} e\n{# note #}\nf"),
  ("loop", "{% for x in xs %}{{ loop.index0 }}{{ x }}{% if loop.first %}F{% endif %}{% if loop.last %}L{% endif %}{% if loop.previtem is defined %}<{{ loop.previtem }}{% endif %},{% endfor %}"),
  ("loop else and filter", "{% for x in [] %}x{% else %}empty{% endfor %}|{% for x in xs if x != 'b' %}{{ x }}{% endfor %}"),
  ("scope", "{% set n = 1 %}{% for x in xs %}{% set n = n + 1 %}{% endfor %}{{ n }}|{% set ns = namespace(n=1) %}{% for x in xs %}{% set ns.n = ns.n + 1 %}{% endfor %}{{ ns.n }}"),
  ("break continue", "{% for i in range(10) %}{% if i == 1 %}{% continue %}{% endif %}{% if i == 4 %}{% break %}{% endif %}{{ i }}{% endfor %}"),
  ("macro", "{% macro greet(name, punct='!') %}Hi {{ name }}{{ punct }}{% endmacro %}{{ greet('Ann') }} {{ greet('Bo', punct='?') }}"),
  ("set block", "{% set body %}[{{ xs | join('-') }}]{% endset %}{{ body | upper }}"),
  ("python values", "{{ none }} {{ true }} {{ [1, 'a', none] }} {{ {'k': 'v'} }} {{ 1.0 }} {{ 7 / 2 }} {{ 7 // 2 }} {{ -7 % 3 }} {{ 'ab' * 2 }}"),
  ("strings", "{{ '  x y  '.strip() }}|{{ 'a,b,,c'.split(',') }}|{{ ' a  b '.split() }}|{{ 'xxhixx'.strip('x') }}|{{ 'Hello'.startswith('He') }}|{{ 'abc'[::-1] }}|{{ 'abcdef'[1:4] }}|{{ 'abc' ~ 1 }}"),
  ("tests", "{{ x is defined }}{{ missing is defined }}{{ none is none }}{{ 'a' is string }}{{ {} is mapping }}{{ [] is sequence }}{{ 1 is number }}{{ missing is iterable }}"),
  ("filters", "{{ missing | default('d') }}|{{ '' | default('e', true) }}|{{ {'b': 1, 'A': 2} | dictsort }}|{{ xs | length }}|{{ xs | map('upper') | list }}|{{ {'a': [1, 'é']} | tojson }}|{{ {'a': 1} | tojson(indent=2) }}"),
  ("item and attribute", "{{ d.k }}{{ d['k'] }}{{ d.get('z', 'none') }}{% for k, v in d.items() %}{{ k }}={{ v }}{% endfor %}{{ d.missing is defined }}"),
  ("conditional expression", "{{ 'y' if xs else 'n' }}{{ 'y' if [] else 'n' }}{{ 'a' in 'cat' }}{{ 'b' not in xs }}{{ 2 in [1, 2] }}"),
]
out = []
for name, source in cases:
    text = env.from_string(source).render(xs=["a", "b", "c"], x=1, d={"k": "v"})
    out.append({"name": name, "template": source, "text": text})
print(json.dumps(out, ensure_ascii=False, indent=1))
