const fs = require('fs');
const path = require('path');

const root = process.cwd();
const dir = path.join(root, '.understand-anything');
const intermediate = path.join(dir, 'intermediate');
const scan = JSON.parse(fs.readFileSync(path.join(intermediate, 'scan-result.json'), 'utf8'));
const assembled = JSON.parse(fs.readFileSync(path.join(intermediate, 'assembled-graph.json'), 'utf8'));
const commit = process.argv[2];
const analyzedAt = new Date().toISOString();

const fileLevel = new Set(['file', 'config', 'document', 'service', 'pipeline', 'table', 'schema', 'resource', 'endpoint']);
const nodesById = new Map(assembled.nodes.map(n => [n.id, n]));
const fileNodes = assembled.nodes.filter(n => fileLevel.has(n.type));

function layerFor(node) {
  const p = node.filePath || '';
  if (p.startsWith('phases/')) return 'layer:curriculum-lessons';
  if (p.startsWith('site/')) return 'layer:static-site';
  if (p.startsWith('scripts/') || p === 'requirements.txt') return 'layer:automation';
  if (p.startsWith('assets/')) return 'layer:assets';
  if (p.startsWith('.github/') || p.startsWith('.vscode/') || ['vercel.json', '.coderabbit.yaml', '.env'].includes(p)) return 'layer:configuration';
  if (/\.(md|rst|txt)$/i.test(p) || ['AGENTS.md', 'README.md', 'ROADMAP.md', 'CONTRIBUTING.md', 'CHANGELOG.md', 'FORKING.md', 'CODE_OF_CONDUCT.md', 'LICENSE', 'SPONSORS.md'].includes(p)) return 'layer:project-docs';
  return 'layer:configuration';
}

const layerDefs = {
  'layer:curriculum-lessons': ['课程内容', '所有阶段课程、讲义、测验、示例实现、测试和 lesson outputs。'],
  'layer:static-site': ['静态站点', '用于浏览课程目录、课程页面、术语表和学习进度的前端资源与构建脚本。'],
  'layer:automation': ['自动化与审计', '维护课程一致性的 Python 与 shell 工具，包括目录生成、审计、链接检查和脚手架。'],
  'layer:project-docs': ['项目级文档', '面向学习者、贡献者和维护者的仓库入口文档、行为准则和变更记录。'],
  'layer:configuration': ['配置与集成', 'CI、编辑器、部署、环境和服务集成配置。'],
  'layer:assets': ['静态资产', '站点和仓库展示使用的图像、样式相关资产和占位资源。'],
};

const grouped = new Map(Object.keys(layerDefs).map(id => [id, []]));
for (const node of fileNodes) grouped.get(layerFor(node)).push(node.id);
const layers = [...grouped.entries()].map(([id, nodeIds]) => ({
  id,
  name: layerDefs[id][0],
  description: layerDefs[id][1],
  nodeIds: nodeIds.filter(id => nodesById.has(id)),
})).filter(layer => layer.nodeIds.length);

fs.writeFileSync(path.join(intermediate, 'layers.json'), JSON.stringify(layers, null, 2));

const pick = (...ids) => ids.filter(id => nodesById.has(id));
const phaseNodes = fileNodes.filter(n => (n.filePath || '').startsWith('phases/')).slice(0, 12).map(n => n.id);
const scriptNodes = fileNodes.filter(n => (n.filePath || '').startsWith('scripts/')).slice(0, 8).map(n => n.id);
const siteNodes = fileNodes.filter(n => (n.filePath || '').startsWith('site/')).slice(0, 8).map(n => n.id);

const tour = [
  {
    order: 1,
    title: '项目入口与贡献规则',
    description: '先阅读 README、AGENTS 和 ROADMAP，理解这个仓库是课程产品而不是 SaaS 应用，以及课程数量、阶段结构和贡献约束。',
    nodeIds: pick('document:README.md', 'document:AGENTS.md', 'document:ROADMAP.md'),
    languageLesson: '把仓库级文档当作架构入口：这里定义了课程目录、lesson contract 和自动化边界。',
  },
  {
    order: 2,
    title: '课程内容骨架',
    description: '进入 phases/ 查看每个阶段与课程目录；每课通常包含 docs、code、quiz 和 outputs，形成 Build It / Use It 的重复结构。',
    nodeIds: phaseNodes,
    languageLesson: '课程目录本身就是领域模型：phase 表示学习阶段，lesson 表示可验证、可交付的学习单元。',
  },
  {
    order: 3,
    title: '自动化质量闸门',
    description: '查看 scripts/ 下的审计与生成脚本，理解仓库如何保持 lesson frontmatter、quiz schema、README counts 和目录数据一致。',
    nodeIds: scriptNodes,
    languageLesson: '课程仓库的可靠性主要来自脚本化不变量，而不是运行时服务。',
  },
  {
    order: 4,
    title: '静态站点发布面',
    description: '查看 site/ 下的构建脚本和前端文件，理解课程内容如何转化为可浏览的目录、课程页和术语表页面。',
    nodeIds: siteNodes,
    languageLesson: 'site/build.js 是内容仓库和用户可见网站之间的转换层。',
  },
  {
    order: 5,
    title: '术语与可复用产物',
    description: '最后查看 glossary 与 outputs，理解跨课程术语如何统一，以及课程产物如何被索引为 prompt、skill、agent 或 MCP server。',
    nodeIds: pick('document:glossary/terms.md', 'document:glossary/README.md', 'config:outputs/index.json'),
    languageLesson: '大型课程体系需要共享术语层，否则不同课程会漂移出不兼容的解释。',
  },
].map(step => ({ ...step, nodeIds: step.nodeIds.filter(id => nodesById.has(id)) }));

fs.writeFileSync(path.join(intermediate, 'tour.json'), JSON.stringify(tour, null, 2));

const graph = {
  version: '1.0.0',
  project: {
    name: scan.name,
    languages: scan.languages,
    frameworks: scan.frameworks,
    description: scan.description,
    analyzedAt,
    gitCommitHash: commit,
  },
  nodes: assembled.nodes,
  edges: assembled.edges,
  layers,
  tour,
};

fs.writeFileSync(path.join(intermediate, 'assembled-graph.json'), JSON.stringify(graph, null, 2));
console.log(JSON.stringify({
  nodes: graph.nodes.length,
  edges: graph.edges.length,
  layers: graph.layers.length,
  tour: graph.tour.length,
}, null, 2));
