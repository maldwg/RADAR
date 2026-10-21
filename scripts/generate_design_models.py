#!/usr/bin/env python3
"""Build repository-derived UML and SQLite ER models without accessing run data.

Class members come from Python ASTs. Database models come from executing only the
literal SCHEMA constants in disposable, in-memory SQLite connections. UML relationships are inferred from imports,
annotations, assignments and protocol shapes. Focused layouts are presentation
choices; SVG is the primary output, with optional editable source exports. No external diagram service is used.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from html import escape
import json
from pathlib import Path
import re
import sqlite3
import textwrap

ROOT = Path(__file__).resolve().parents[1]
OUT = Path('docs/_static/models')
INK, MUTED, LINE = '#24323a', '#5d6a71', '#657c8a'
BLUE, GREEN, GOLD, PURPLE = '#eaf2f8', '#edf5ee', '#faf3e5', '#f1edf7'


def classes(root: Path) -> dict[str, dict]:
    """Inventory top-level production classes, including duplicate short names."""
    result = {}
    for path in sorted((root / 'src/radar').rglob('*.py')):
        module = '.'.join(path.relative_to(root / 'src').with_suffix('').parts)
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            fields = [(n.target.id, ast.unparse(n.annotation)) for n in node.body
                      if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)]
            methods = [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                       and (not n.name.startswith('_') or n.name == '__init__')]
            bases = [ast.unparse(base) for base in node.bases]
            decorators = [ast.unparse(item) for item in node.decorator_list]
            result[f'{module}.{node.name}'] = {
                'name': node.name, 'module': module, 'path': str(path.relative_to(root)),
                'line': node.lineno, 'fields': fields, 'methods': methods, 'bases': bases,
                'kind': 'protocol' if 'Protocol' in bases else
                        'dataclass' if any(item.startswith('dataclass') for item in decorators) else 'class',
            }
    return result


def schema(root: Path, name: str) -> dict:
    """Inspect the real SQL DDL, never a live cache or reporting database."""
    tree = ast.parse((root / f'src/radar/common/{name}.py').read_text(encoding='utf-8'))
    ddl = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == 'SCHEMA' for target in node.targets))
    db = sqlite3.connect(':memory:')
    try:
        db.executescript(ddl)
        tables = {}
        for (table,) in db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
            columns = [{'name': n, 'type': t, 'not_null': bool(nn), 'pk': pk}
                       for _, n, t, nn, _, pk in db.execute(f'PRAGMA table_info("{table}")')]
            foreign_keys = [{'id': i, 'sequence': seq, 'table': target, 'from': local, 'to': remote,
                             'nullable': not next(c['not_null'] for c in columns if c['name'] == local)}
                            for i, seq, target, local, remote, *_ in db.execute(f'PRAGMA foreign_key_list("{table}")')]
            tables[table] = {'columns': columns, 'foreign_keys': foreign_keys}
        return tables
    finally:
        db.close()


def q(module: str, name: str) -> str:
    """Construct a fully qualified RADAR class name."""
    return f'radar.{module}.{name}'


D = 'adapters.contracts'
DISC, LOOK, EVID, REF, CAT = [q(D, n) for n in ['DiscoveryAdapter','LookupAdapter','EvidenceAdapter','ReferenceAdapter','CatalogueAdapter']]
GH, OA, SS, LOCAL = [q('adapters.' + m, n) for m,n in [('github','GitHubAdapter'),('openalex','OpenAlexAdapter'),('semantic_scholar','SemanticScholarAdapter'),('local','LocalDiscoveryAdapter')]]
GHC, OAC, HTTP = q('github.collect','GitHubClient'),q('openalex.collect','OpenAlexClient'),q('adapters.http','JsonTransport')
PES, BLOCKS, BLOCK = q('common.paper_evidence','PaperEvidenceService'),q('common.evidence_blocks','EvidenceBlockIndex'),q('common.evidence_blocks','EvidenceBlock')
QUOTE, QUOTATION = q('common.evidence_quotes','SourceQuoteIndex'),q('common.evidence_quotes','SourceQuotation')
LSA, CONTRACT, STUDY, PROTOCOL = q('common.sources','LocalSourceAdapter'),q('common.sources','SourceContract'),q('common.study_screening','StudyScreeningAdapter'),q('common.study_screening','StudyProtocol')
OLLAMA, SECTION = q('common.ollama','OllamaClient'),q('common.ollama','EvidenceSection')
NLI, TRANS, ONLI = q('common.nli','NLIBackend'),q('common.nli','TransformersNLIBackend'),q('common.llm_nli','OllamaNLIBackend')
RUNTIME, LLMCONF = q('common.nli','NLIRuntimeConfig'),q('common.llm_nli','LLMNLIConfig')
GENERATORS = [q(module, 'Generator') for module in ['github.pipeline','github.reconcile','openalex.pipeline','openalex.reconcile']]

def relationships(root: Path, inventory: dict) -> tuple[list, dict]:
    """Infer UML edges from imports, annotations, assignments and class references.

    Protocol matches check member presence and accepted call shapes. They are
    static candidates, not a substitute for checking variance or executing code.
    Every edge retains source locations describing why it was inferred.
    """
    import inspect

    modules = {}
    nodes = {}
    for path in sorted((root/'src/radar').rglob('*.py')):
        module = '.'.join(path.relative_to(root/'src').with_suffix('').parts)
        tree = ast.parse(path.read_text(encoding='utf-8'))
        imports = {}
        for item in ast.walk(tree):
            if isinstance(item, ast.Import):
                for alias in item.names:
                    imports[alias.asname or alias.name.split('.')[0]] = alias.name if alias.asname else alias.name.split('.')[0]
            elif isinstance(item, ast.ImportFrom):
                prefix = item.module or ''
                if item.level:
                    parts = module.split('.')[:-item.level]
                    prefix = '.'.join([*parts, prefix]).rstrip('.')
                for alias in item.names:
                    imports[alias.asname or alias.name] = f'{prefix}.{alias.name}'
        modules[module] = (tree, imports, str(path.relative_to(root)))
        for item in tree.body:
            if isinstance(item, ast.ClassDef): nodes[f'{module}.{item.name}'] = item

    def resolve(node, module):
        """Resolve imported aliases and qualified names without importing modules."""
        if isinstance(node, ast.Name):
            local = f'{module}.{node.id}'
            return local if local in inventory else modules[module][1].get(node.id, local)
        if isinstance(node, ast.Attribute):
            return f'{resolve(node.value,module)}.{node.attr}'
        return ''

    def references(node, module):
        """Find repository classes inside an annotation, including forward strings."""
        if node is None: return set()
        if isinstance(node, ast.Constant) and isinstance(node.value,str):
            try: return references(ast.parse(node.value,mode='eval').body,module)
            except SyntaxError: return set()
        result = set()
        for child in ast.walk(node):
            name = resolve(child,module)
            if name in inventory: result.add(name)
            if isinstance(child,ast.Constant) and isinstance(child.value,str):
                result.update(references(child,module))
        return result

    returns = {}
    for module,(tree,_,_) in modules.items():
        for item in tree.body:
            if isinstance(item,(ast.FunctionDef,ast.AsyncFunctionDef)):
                returns[f'{module}.{item.name}'] = references(item.returns,module)
            elif isinstance(item,ast.ClassDef):
                for method in item.body:
                    if isinstance(method,(ast.FunctionDef,ast.AsyncFunctionDef)):
                        returns[f'{module}.{item.name}.{method.name}'] = references(method.returns,module)

    def inferred(value, module, env):
        """Infer stored types from parameters, constructors and annotated factories."""
        if value is None: return set()
        if isinstance(value,ast.Name): return env.get(value.id,set())
        result = set()
        for item in ast.walk(value):
            if isinstance(item,ast.Call):
                name = resolve(item.func,module)
                if name in inventory: result.add(name)
                result.update(returns.get(name,set()))
        if isinstance(value,ast.BoolOp):
            for item in value.values: result.update(inferred(item,module,env))
        if isinstance(value,ast.IfExp):
            result.update(inferred(value.body,module,env));result.update(inferred(value.orelse,module,env))
        return result

    module_env = {}
    for module,(tree,_,_) in modules.items():
        env = {}
        for item in tree.body:
            if isinstance(item,ast.Assign):
                for target in item.targets:
                    if isinstance(target,ast.Name): env[target.id] = inferred(item.value,module,env)
            elif isinstance(item,ast.AnnAssign) and isinstance(item.target,ast.Name):
                env[item.target.id] = references(item.annotation,module) | inferred(item.value,module,env)
        module_env[module] = env

    found = defaultdict(set)
    attributes = defaultdict(set)
    methods = {}

    def edge(source, target, kind, label, node):
        """Retain source evidence for each unique non-self relationship."""
        if source == target or target not in inventory: return
        key = (source,target,kind,label)
        found[key].add(f'{inventory[source]["path"]}:{node.lineno}')

    def assignment(source, target, value, module, env, node, annotation=None):
        """Handle direct and tuple-unpacked assignments to names and self fields."""
        if isinstance(target,(ast.Tuple,ast.List)):
            if isinstance(value,(ast.Tuple,ast.List)) and len(target.elts)==len(value.elts):
                for left,right in zip(target.elts,value.elts): assignment(source,left,right,module,env,node)
            return
        types = references(annotation,module) | inferred(value,module,env)
        if isinstance(target,ast.Name): env[target.id] = types
        if isinstance(target,ast.Attribute) and isinstance(target.value,ast.Name) and target.value.id in {'self','cls'}:
            attributes[source].add(target.attr)
            for name in types: edge(source,name,'association',target.attr,node)

    for source,cls in nodes.items():
        module = inventory[source]['module']
        methods[source] = {item.name:item for item in cls.body if isinstance(item,(ast.FunctionDef,ast.AsyncFunctionDef))}
        env = dict(module_env[module])
        for base in cls.bases:
            edge(source,resolve(base,module),'inheritance','extends',base)
        for item in cls.body:
            if isinstance(item,ast.AnnAssign) and isinstance(item.target,ast.Name):
                attributes[source].add(item.target.id)
                for name in references(item.annotation,module):
                    callable_field = any(isinstance(n,ast.Name) and n.id=='Callable' for n in ast.walk(item.annotation))
                    edge(source,name,'dependency' if callable_field else 'association',item.target.id,item)
            elif isinstance(item,ast.Assign):
                for target in item.targets:
                    if isinstance(target,ast.Name):
                        attributes[source].add(target.id)
                        for name in inferred(item.value,module,env): edge(source,name,'association',target.id,item)
        for method in methods[source].values():
            local = dict(env)
            args = [*method.args.posonlyargs,*method.args.args,*method.args.kwonlyargs]
            for arg in args:
                local[arg.arg] = references(arg.annotation,module)
                for name in local[arg.arg]: edge(source,name,'dependency',f'{method.name} parameter',arg)
            for name in references(method.returns,module): edge(source,name,'dependency',f'{method.name} returns',method)
            for item in ast.walk(method):
                if isinstance(item,ast.Assign):
                    for target in item.targets: assignment(source,target,item.value,module,local,item)
                elif isinstance(item,ast.AnnAssign):
                    assignment(source,item.target,item.value,module,local,item,item.annotation)
                elif isinstance(item,ast.Call):
                    target = resolve(item.func,module)
                    if target in inventory: edge(source,target,'dependency','constructs',item)
                    elif isinstance(item.func,ast.Attribute):
                        edge(source,resolve(item.func.value,module),'dependency','uses',item)

    def members(source, visited=None):
        """Include inherited methods and attributes when comparing protocol shapes."""
        visited = set() if visited is None else visited
        if source in visited: return {},set()
        visited.add(source)
        inherited_methods = {}; inherited_attributes = set()
        for base in nodes[source].bases:
            parent = resolve(base,inventory[source]['module'])
            if parent in nodes:
                parent_methods,parent_attributes = members(parent,visited)
                inherited_methods.update(parent_methods);inherited_attributes.update(parent_attributes)
        inherited_methods.update(methods[source]);inherited_attributes.update(attributes[source])
        return inherited_methods,inherited_attributes

    def signature(method):
        """Build a call signature without evaluating annotations or defaults."""
        args = method.args
        positional = [*args.posonlyargs,*args.args]
        defaults = len(positional)-len(args.defaults)
        parameters = []
        for index,arg in enumerate(positional):
            if index==0 and arg.arg in {'self','cls'}: continue
            kind = inspect.Parameter.POSITIONAL_ONLY if index<len(args.posonlyargs) else inspect.Parameter.POSITIONAL_OR_KEYWORD
            default = inspect.Parameter.empty if index<defaults else None
            parameters.append(inspect.Parameter(arg.arg,kind,default=default))
        if args.vararg: parameters.append(inspect.Parameter(args.vararg.arg,inspect.Parameter.VAR_POSITIONAL))
        for arg,default in zip(args.kwonlyargs,args.kw_defaults):
            parameters.append(inspect.Parameter(arg.arg,inspect.Parameter.KEYWORD_ONLY,default=inspect.Parameter.empty if default is None else None))
        if args.kwarg: parameters.append(inspect.Parameter(args.kwarg.arg,inspect.Parameter.VAR_KEYWORD))
        return inspect.Signature(parameters)

    def compatible(required, implementation):
        """Check minimal/full positional and keyword calls permitted by a protocol."""
        required = signature(required);implementation = signature(implementation)
        P = inspect.Parameter
        params = list(required.parameters.values())
        if any(p.kind in {P.VAR_POSITIONAL,P.VAR_KEYWORD} for p in params): return False
        positional = [p for p in params if p.kind in {P.POSITIONAL_ONLY,P.POSITIONAL_OR_KEYWORD}]
        kwonly = [p for p in params if p.kind==P.KEYWORD_ONLY]
        calls = [
            ([None for p in positional],{p.name:None for p in kwonly}),
            ([None for p in positional if p.default is P.empty],{p.name:None for p in kwonly if p.default is P.empty}),
            ([None for p in positional if p.kind==P.POSITIONAL_ONLY],{p.name:None for p in params if p.kind!=P.POSITIONAL_ONLY}),
        ]
        for args,kwargs in calls:
            try: implementation.bind(*args,**kwargs)
            except TypeError: return False
        return True

    for source in nodes:
        if inventory[source]['kind']=='protocol': continue
        supplied_methods,supplied_attributes = members(source)
        for target in nodes:
            if inventory[target]['kind']!='protocol': continue
            needed_methods,needed_attributes = members(target)
            needed_methods = {name:node for name,node in needed_methods.items() if not name.startswith('_')}
            if not needed_methods or not needed_attributes <= supplied_attributes: continue
            if not needed_methods.keys() <= supplied_methods.keys(): continue
            if all(compatible(method,supplied_methods[name]) for name,method in needed_methods.items()):
                edge(source,target,'realization','structural candidate',nodes[source])

    # Collapse multiple reasons between the same classes and kind into one edge;
    # full source locations remain available in the machine-readable inventory.
    groups = defaultdict(dict)
    for (source,target,kind,label),locations in found.items():
        groups[(source,target,kind)].setdefault(label,set()).update(locations)
    relations = []; evidence = {}
    for (source,target,kind),labels in sorted(groups.items()):
        if kind=='dependency' and (source,target,'association') in groups: continue
        label = ', '.join(sorted(labels))
        relations.append((source,target,kind,label))
        evidence['|'.join((source,target,kind))] = {key:sorted(value) for key,value in sorted(labels.items())}
    return relations,evidence


VIEWS = [
    ('uml-discovery','Provider discovery and HTTP clients',
     [[DISC], [GH,OA,SS,LOCAL], [GHC,OAC,HTTP]]),
    ('uml-capabilities','Optional provider capabilities',
     [[LOOK,EVID,REF,CAT], [OA,SS,GH], [PES]]),
    ('uml-records','Discovery configuration and normalized records',
     [[q('common.discovery','DiscoverySource'),q('common.paper_collection','PaperSearchClient'),q('adapters.registry','ProviderRegistration')],
      [DISC,q(D,'ProviderDescriptor')], [q(D,'ProviderQuery'),q(D,'SearchPage'),q(D,'SourceRecord')]]),
    ('uml-screening','Scientific screening and evidence grounding',
     [[LSA,q('common.evaluation','RatingProfile')], [STUDY,PROTOCOL,CONTRACT],
      [BLOCKS,BLOCK], [QUOTE,QUOTATION]]),
    ('uml-generators','Model generation and repository evidence contracts',
     [GENERATORS, [OLLAMA,q('github.enrich','GitHubEvidenceClient'),q('github.pipeline','EvidenceEnricher')]]),
    ('uml-nli','NLI backends, configuration and score types',
     [[NLI,SECTION], [TRANS,ONLI,q('common.nli','NLIProfile')],
      [RUNTIME,LLMCONF,q('common.nli','NLIField')], [q('common.nli','EvidenceChunk'),q('common.nli','NLIScore'),q('common.cache','PipelineCache')]]),
]


def model_views(inventory: dict) -> list:
    """Keep every class visible, including small results, helpers and errors."""
    views = []
    for name,title,rows in VIEWS:
        rows = [[key for key in row if key in inventory] for row in rows]
        rows = [row for row in rows if row]
        if rows: views.append((name,title,rows))
    covered = {node for _,_,rows in views for row in rows for node in row}
    rest = sorted(set(inventory) - covered)
    # Keep remaining value/support types in compact, paginated class views.
    for index in range(0,len(rest),12):
        page = rest[index:index+12]
        views.append((f'uml-support-{index//12+1}',f'Support classes and result types ({index//12+1})',
                      [page[n:n+3] for n in range(0,len(page),3)]))
    return views


def identifier(value: str) -> str:
    """Encode a qualified name as a diagram-safe identifier."""
    return re.sub(r'\W','_',value)


def plantuml(inventory: dict, nodes: list[str], relations: list) -> str:
    """Export class members and UML relationship kinds in editable PlantUML."""
    lines = ['@startuml','hide empty members','skinparam classAttributeIconSize 0']
    for key in nodes:
        c = inventory[key]
        stereo = f' <<{c["kind"]}>>' if c['kind'] != 'class' else ''
        lines.append(f'class "{key}" as {identifier(key)}{stereo} {{')
        lines += [f'  +{name}: {kind}' for name,kind in c['fields']]
        lines += [f'  +{name}()' for name in c['methods']]
        lines.append('}')
    external_bases = {'ValueError':'builtins.ValueError', 'RuntimeError':'builtins.RuntimeError', 'logging.Formatter':'logging.Formatter'}
    emitted = set()
    for key in nodes:
        for base in inventory[key]['bases']:
            if base not in external_bases: continue
            external = external_bases[base]
            if external not in emitted:
                lines.append(f'class \"{external}\" as {identifier(external)} <<external>>')
                emitted.add(external)
            lines.append(f'{identifier(key)} --|> {identifier(external)}')
    arrows = {'realization':'..|>','inheritance':'--|>','association':'-->','dependency':'..>'}
    for source,target,kind,label in relations:
        if source in nodes and target in nodes:
            lines.append(f'{identifier(source)} {arrows[kind]} {identifier(target)} : {label}')
    return '\n'.join(lines + ['@enduml',''])


def er_mermaid(tables: dict) -> str:
    """Export actual tables and foreign-key cardinalities as a Mermaid ER model."""
    lines = ['erDiagram']
    for name, table in tables.items():
        lines.append(f'    {name} {{')
        for column in table['columns']:
            keys = []
            if column['pk']: keys.append('PK')
            if any(fk['from'] == column['name'] for fk in table['foreign_keys']): keys.append('FK')
            note = 'NOT NULL' if column['not_null'] else 'nullable' if not column['pk'] else 'primary key'
            lines.append(f'        {column["type"]} {column["name"]} {",".join(keys)} "{note}"')
        lines.append('    }')
    for child,table in tables.items():
        for fk in table['foreign_keys']:
            parent = fk['table']; cardinality = '|o' if fk['nullable'] else '||'
            # Non-identifying relationship: the FK is not universally a child PK.
            identifying = next(c['pk'] for c in table['columns'] if c['name'] == fk['from'])
            connector = '--' if identifying else '..'
            lines.append(f'    {parent} {cardinality}{connector}o{{ {child} : "{fk["from"]}"')
    return '\n'.join(lines) + '\n'


def text(x: float,y: float,value: str,*,size=13,fill=INK,weight='normal',anchor='start') -> str:
    """Escape and position an accessible SVG text element."""
    return f'<text x="{x:g}" y="{y:g}" fill="{fill}" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}">{escape(value)}</text>'


def class_card(key: str,c: dict) -> dict:
    """Select compact UML members while preserving module and base information."""
    lines = [f'+ {name}: {kind}' for name,kind in c['fields'][:3]]
    if len(c['fields']) > 3: lines.append(f'… {len(c["fields"])-3} more attributes')
    if c['bases'] and c['kind'] != 'protocol': lines.append('base: ' + ', '.join(c['bases']))
    split = len(lines)
    visible_methods = [name for name in c['methods'] if name != '__init__'][:4]
    lines += [f'+ {name}()' for name in visible_methods]
    if len(c['methods']) > len(visible_methods): lines.append(f'… {len(c["methods"])-len(visible_methods)} more methods (incl. constructor)')
    return {'key':key,'title':c['name'],'subtitle':c['module'].removeprefix('radar.'),
            'stereotype':c['kind'],'lines':lines,'split':split,'color':BLUE if c['kind']=='protocol' else GREEN if c['kind']=='dataclass' else '#f6f7f8'}


def er_card(key: str,table: dict) -> dict:
    """Describe a table with every column and its actual SQL key flags."""
    lines=[]
    for col in table['columns']:
        flags=[]
        if col['pk']: flags.append(f'PK{col["pk"]}')
        if any(fk['from']==col['name'] for fk in table['foreign_keys']): flags.append('FK')
        if col['not_null']: flags.append('NN')
        lines.append(f'{col["name"]}: {col["type"]}' + (f' [{",".join(flags)}]' if flags else ''))
    return {'key':key,'title':key,'subtitle':'SQLite table','stereotype':'table',
            'lines':lines,'split':0,'color':PURPLE}


def render_svg(title: str,subtitle: str,rows: list[list[str]],cards: dict,edges: list,*,er=False) -> str:
    """Render measured class/table compartments and orthogonal relationship lanes.

    A small fixed layout keeps the exported diagrams independent of Graphviz or a
    hosted renderer. Text is wrapped, never discarded; full UML members are in the generated class inventory.
    """
    width_card,step,margin = 370,430,50
    ncols=max(map(len,rows)); corewidth=max(1100,ncols*step+40)
    # Assign one routing lane per edge. Long edges go around the right boundary,
    # never through intervening class/table boxes.
    rightlane=corewidth+30
    width=rightlane+max(1,len(edges))*10+60
    layout={}; row_bottom={}; y=155
    for rank,row in enumerate(rows):
        maxheight=0
        for col,key in enumerate(row):
            card=cards[key]
            body=[]
            for index,line in enumerate(card['lines']):
                wrapped=textwrap.wrap(line,width=47,break_long_words=True,break_on_hyphens=False) or ['']
                body += [(segment,index==card['split'] and part==0 and card['split']>0)
                         for part,segment in enumerate(wrapped)]
            height=94+len(body)*19
            x=margin+(corewidth-2*margin-len(row)*step+60)/2+col*step
            layout[key]=(x,y,width_card,height,rank,body)
            maxheight=max(maxheight,height)
        row_bottom[rank]=y+maxheight
        y+=maxheight+max(140, len(edges)*24+60)
    last_row_lanes=[index*24 for index,edge in enumerate(edges)
                    if layout[edge[0]][4] == layout[edge[1]][4] == len(rows)-1]
    height=max(y0+h for _,y0,_,h,_,_ in layout.values())+100+max(last_row_lanes,default=0)
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:g}" height="{height:g}" viewBox="0 0 {width:g} {height:g}" role="img" aria-labelledby="title description">',
         f'<title id="title">{escape(title)}</title><desc id="description">{escape(subtitle)}. See adjacent editable sources and documentation for complete relationships.</desc>',
         '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M 1 1 L 9 5 L 1 9" fill="none" stroke="#657c8a" stroke-width="1.5"/></marker>',
         '<marker id="many" viewBox="0 0 24 20" refX="23" refY="10" markerWidth="24" markerHeight="20" markerUnits="userSpaceOnUse" orient="auto"><circle cx="4" cy="10" r="3" fill="white" stroke="#657c8a"/><path d="M 8 10 H 23 M 12 10 L 23 2 M 12 10 L 23 18" fill="none" stroke="#657c8a"/></marker>',
         '<marker id="triangle" viewBox="0 0 12 12" refX="11" refY="6" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1 1 L 11 6 L 1 11 Z" fill="white" stroke="#657c8a" stroke-width="1.2"/></marker></defs>',
         '<rect width="100%" height="100%" fill="white"/>',
         '<g font-family="DejaVu Sans, sans-serif">',text(40,38,title,size=25,weight='bold'),text(40,66,subtitle,size=13,fill=MUTED)]
    legend='PKn = primary-key order; FK = foreign key; NN = NOT NULL. Parent/child cardinalities are printed on each relationship.' if er else 'Hollow triangle: inheritance (solid) or protocol realization (dashed). Arrow: association (solid) or use/return dependency (dashed).'
    svg += [text(40,94,legend,size=12,fill=MUTED)]
    # Draw connectors before cards so card backgrounds mask any terminal overlap.
    for index,(source,target,kind,label) in enumerate(edges):
        label=textwrap.shorten(label,width=70,placeholder='…')
        sx,sy,sw,sh,srank,_=layout[source];tx,ty,tw,th,trank,_=layout[target]
        dash=' stroke-dasharray="6 4"' if kind in {'dependency','realization'} else ''
        marker='many' if er else 'triangle' if kind in {'realization','inheritance'} else 'arrow'
        lane=rightlane+index*10
        label_anchor='middle'
        if srank == trank:
            # Route in the lower gutter, with vertical terminals on distinct ports.
            start=(sx+sw*.65,sy+sh);end=(tx+tw*.35,ty+th)
            mid=row_bottom[srank]+30+index*24
            path=f'M {start[0]} {start[1]} V {mid} H {end[0]} V {end[1]}'
            lx=(start[0]+end[0])/2;ly=mid-5
        elif abs(srank-trank)==1:
            down=srank<trank
            start=(sx+sw*(.2+.6*((index%5)/4)),sy+sh if down else sy)
            end=(tx+tw*(.2+.6*(((index+2)%5)/4)),ty if down else ty+th)
            mid=row_bottom[min(srank,trank)]+30+index*24
            path=f'M {start[0]} {start[1]} V {mid} H {end[0]} V {end[1]}'
            lx=(start[0]+end[0])/2;ly=mid-5
        else:
            down=srank<trank
            start=(sx+sw*.8,sy+sh if down else sy)
            end=(tx+tw*.8,ty if down else ty+th)
            exit_y=row_bottom[srank]+30+index*24 if down else sy-30-index*24
            entry_y=ty-30-index*24 if down else row_bottom[trank]+30+index*24
            path=f'M {start[0]} {start[1]} V {exit_y} H {lane} V {entry_y} H {end[0]} V {end[1]}'
            lx=(start[0]+lane)/2;ly=exit_y-5
        svg.append(f'<path d="{path}" fill="none" stroke="{LINE}" stroke-width="1.3"{dash} marker-end="url(#{marker})"/>')
        # White underpaint makes labels legible across unrelated connectors.
        svg.append(f'<text x="{lx:g}" y="{ly:g}" font-size="12" text-anchor="{label_anchor}" fill="white" stroke="white" stroke-width="4">{escape(label)}</text>')
        svg.append(text(lx,ly,label,size=12,anchor=label_anchor))
    for key,(x,y,w,h,_,body) in layout.items():
        card=cards[key]
        svg += [f'<g><title>{escape(key)}</title>',f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="white" stroke="{LINE}"/>',
                f'<path d="M {x+4} {y} H {x+w-4} Q {x+w} {y} {x+w} {y+4} V {y+77} H {x} V {y+4} Q {x} {y} {x+4} {y}" fill="{card["color"]}"/>',
                text(x+12,y+19,f'«{card["stereotype"]}»',size=11,fill=MUTED),
                text(x+12,y+42,card['title'],size=17,weight='bold'),
                text(x+12,y+63,card['subtitle'],size=11,fill=MUTED),
                f'<path d="M {x} {y+77} H {x+w}" stroke="{LINE}"/>']
        for n,(line,separator) in enumerate(body):
            if separator: svg.append(f'<path d="M {x+8} {y+82+n*19} H {x+w-8}" stroke="#d7dfe3"/>')
            svg.append(text(x+12,y+98+n*19,line,size=12))
        svg.append('</g>')
    svg += [text(40,height-16,'RADAR • repository-derived model • editable source and complete member inventory accompany this figure',size=11,fill=MUTED),'</g></svg>']
    return '\n'.join(svg)+'\n'


def render_overview(inventory: dict, relations: list, views: list) -> str:
    """Render every class and inferred edge in a compact, linked SVG overview."""
    width, height = 1810, 190+((len(inventory)+4)//5)*100
    layout = {}
    for index,(key,cls) in enumerate(sorted(inventory.items())):
        layout[key] = (35+(index%5)*350,150+(index//5)*100)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
           '<title id="title">RADAR — complete class diagram generated from Python</title>',
           '<desc id="desc">All production classes and statically inferred inheritance, association, dependency and protocol candidate relationships. Each class links to a focused SVG.</desc>',
           '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M 1 1 L 9 5 L 1 9" fill="none" stroke="#657c8a"/></marker>',
           '<marker id="triangle" viewBox="0 0 12 12" refX="11" refY="6" markerWidth="10" markerHeight="10" orient="auto"><path d="M 1 1 L 11 6 L 1 11 Z" fill="white" stroke="#657c8a"/></marker></defs>',
           '<rect width="100%" height="100%" fill="white"/><g font-family="DejaVu Sans, sans-serif">',
           text(35,40,'RADAR — complete class diagram',size=26,weight='bold'),
           text(35,68,f'{len(inventory)} classes • {len(relations)} relationships inferred from Python source • click a class to open its focused SVG',size=14,fill=MUTED),
           text(35,95,'Solid triangle: inheritance. Dashed triangle: protocol candidate. Solid arrow: stored association. Dashed arrow: dependency.',size=13,fill=MUTED)]
    for source,target,kind,label in relations:
        sx,sy=layout[source];tx,ty=layout[target]
        if sy==ty:
            start=(sx+320 if sx<tx else sx,sy+40);end=(tx if sx<tx else tx+320,ty+40)
            bend=28+abs(tx-sx)*.08
            path=f'M {start[0]} {start[1]} C {start[0]} {start[1]+bend}, {end[0]} {end[1]+bend}, {end[0]} {end[1]}'
        else:
            down=sy<ty
            start=(sx+160,sy+76 if down else sy);end=(tx+160,ty if down else ty+76)
            middle=(start[1]+end[1])/2
            path=f'M {start[0]} {start[1]} C {start[0]} {middle}, {end[0]} {middle}, {end[0]} {end[1]}'
        dashed=' stroke-dasharray="5 4"' if kind in {'realization','dependency'} else ''
        marker='triangle' if kind in {'realization','inheritance'} else 'arrow'
        color='#327350' if kind=='realization' else '#97621d' if kind=='inheritance' else LINE
        svg.append(f'<g><title>{escape(source+" → "+target+": "+kind+"; "+label)}</title><path d="{path}" fill="none" stroke="{color}" opacity="0.5" stroke-width="1.3"{dashed} marker-end="url(#{marker})"/></g>')
    for key,cls in sorted(inventory.items()):
        x,y=layout[key]
        target=next(name for name,_,rows in views if any(key in row for row in rows))
        color=BLUE if cls['kind']=='protocol' else GREEN if cls['kind']=='dataclass' else '#f6f7f8'
        svg += [f'<a href="{target}.svg"><g><title>{escape(key)} — open focused class diagram</title>',
                f'<rect x="{x}" y="{y}" width="320" height="76" rx="4" fill="{color}" stroke="{LINE}"/>',
                text(x+10,y+17,'«'+cls['kind']+'»',size=10,fill=MUTED),
                text(x+10,y+39,cls['name'],size=15,weight='bold'),
                text(x+10,y+60,cls['module'].removeprefix('radar.'),size=10,fill=MUTED),'</g></a>']
    svg += [text(35,height-20,'Static analysis cannot resolve every dynamic factory or prove full protocol type compatibility; see the accompanying documentation.',size=12,fill=MUTED),'</g></svg>']
    return '\n'.join(svg)+'\n'


def er_rows(tables: dict) -> list[list[str]]:
    """Layer the acyclic RADAR schemas by foreign-key dependency."""
    ranks={}
    def rank(name: str) -> int:
        """Resolve the dependency depth of a table."""
        if name not in ranks:
            parents=[fk['table'] for fk in tables[name]['foreign_keys']]
            ranks[name]=max((rank(parent)+1 for parent in parents),default=0)
        return ranks[name]
    groups=defaultdict(list)
    for name in tables: groups[rank(name)].append(name)
    return [groups[n] for n in sorted(groups)]


def write_figure(output: Path,name: str,svg: str,*,pdf: bool) -> None:
    """Write a vector figure and optionally convert it to a portable PDF."""
    (output/f'{name}.svg').write_text(svg,encoding='utf-8')
    if pdf:
        import cairosvg
        cairosvg.svg2pdf(bytestring=svg.encode('utf-8'),write_to=str(output/f'{name}.pdf'))


def generate(root: Path,output: Path,*,pdf: bool=True) -> dict:
    """Generate all diagrams, editable sources and implementation inventories."""
    output.mkdir(parents=True,exist_ok=True)
    inventory=classes(root)
    views=model_views(inventory)
    relations,evidence=relationships(root,inventory)
    for source,target,kind,label in relations:
        if source not in inventory or target not in inventory:
            raise ValueError(f'unknown UML endpoint: {source} -> {target}')
    manifest={'classes':inventory,'views':{},'relations':relations,'relationship_evidence':evidence,'databases':{}}
    for name,title,rows in views:
        nodes=[key for row in rows for key in row]
        edges=[edge for edge in relations if edge[0] in nodes and edge[1] in nodes]
        cards={key:class_card(key,inventory[key]) for key in nodes}
        svg=render_svg(title,'Generated from Python source; selected members shown. Protocol arrows are static structural candidates.',rows,cards,edges)
        write_figure(output,name,svg,pdf=pdf)
        (output/f'{name}.puml').write_text(plantuml(inventory,nodes,edges),encoding='utf-8')
        manifest['views'][name]={'title':title,'classes':nodes}
    write_figure(output,'uml-all-classes',render_overview(inventory,relations,views),pdf=pdf)
    (output/'uml-all-classes.puml').write_text(plantuml(inventory,list(inventory),relations),encoding='utf-8')
    # This HTML-free RST inventory is included in the Sphinx documentation.
    rst=['.. This inventory is generated by scripts/generate_design_models.py.','', 'Complete class inventory','------------------------','',
         f'All {len(inventory)} top-level classes under ``src/radar`` are listed below. '
         'Functions and external dependency classes are outside this inventory. '
         'Methods shown are public methods plus constructors; annotated class fields include dataclass fields. '
         'Instance attributes assigned inside methods are not an exhaustive part of this inventory.','']
    for key,c in inventory.items():
        rst += [f'``{key}``',f'   {c["kind"]}; source: ``{c["path"]}:{c["line"]}``.', '']
        if c['bases']: rst += ['   Bases: '+', '.join(f'``{base}``' for base in c['bases'])+'.','']
        if c['fields']: rst += ['   Fields: '+', '.join(f'``{name}: {kind}``' for name,kind in c['fields'])+'.','']
        if c['methods']: rst += ['   Methods: '+', '.join(f'``{name}()``' for name in c['methods'])+'.','']
    (output/'class_inventory.rst.inc').write_text('\n'.join(rst),encoding='utf-8')
    for name in ['reporting','cache']:
        tables=schema(root,name)
        manifest['databases'][name]=tables
        (output/f'er-{name}.mmd').write_text(er_mermaid(tables),encoding='utf-8')
        edges=[]
        for child,table in tables.items():
            for fk in table['foreign_keys']:
                label=f'{fk["from"]} (parent {"0..1" if fk["nullable"] else "1"}; child 0..*)'
                edges.append((fk['table'],child,'association',label))
        cards={key:er_card(key,table) for key,table in tables.items()}
        svg=render_svg(f'{"Reporting database" if name=="reporting" else "Pipeline cache"} — entity relationship model',
                       f'{len(tables)} actual SQLite tables • {len(edges)} declared foreign keys • no live database files read',
                       er_rows(tables),cards,edges,er=True)
        write_figure(output,f'er-{name}',svg,pdf=pdf)
        if name == 'reporting':
            for suffix, selected in {
                'decisions': ['runs','snapshots','entities','identifiers','source_records','decisions','evidence','stage_memberships','relationships'],
                'evaluation': ['runs','entities','decisions','reference_sets','reference_entities','evaluations','reference_matches'],
            }.items():
                subset={key:table for key,table in tables.items() if key in selected}
                subset={key:{**table,'foreign_keys':[fk for fk in table['foreign_keys'] if fk['table'] in subset]} for key,table in subset.items()}
                subedges=[edge for edge in edges if edge[0] in subset and edge[1] in subset]
                subsvg=render_svg(f'Reporting database — {suffix}', 'Subset of the reporting schema; the complete model includes every table and foreign key.',
                                  er_rows(subset),{key:cards[key] for key in subset},subedges,er=True)
                write_figure(output,f'er-reporting-{suffix}',subsvg,pdf=pdf)
    (output/'model_inventory.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    return manifest


def main() -> None:
    """Regenerate checked-in models or write them to a selected output directory."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/OUT)
    parser.add_argument('--check',action='store_true',help='Check generated SVG/source assets for drift without modifying files or rendering PDFs.')
    parser.add_argument('--no-pdf',action='store_true',help='Generate SVG and editable sources with the standard library only.')
    args=parser.parse_args()
    if args.check:
        import tempfile
        with tempfile.TemporaryDirectory(prefix='radar-model-check-') as tmp:
            generated=Path(tmp)
            generate(ROOT,generated,pdf=False)
            stale=[path.name for path in generated.iterdir()
                   if not (args.output/path.name).is_file() or path.read_bytes()!=(args.output/path.name).read_bytes()]
            if stale:
                parser.exit(1,'Generated models are stale: '+', '.join(sorted(stale))+'\nRun python scripts/generate_design_models.py\n')
        print('Generated SVG and source models are current.')
        return
    result=generate(ROOT,args.output,pdf=not args.no_pdf)
    print(f'Generated {len(result["classes"])} classes, {len(result["views"])} UML views, and 2 database models in {args.output}')


if __name__=='__main__':
    main()
