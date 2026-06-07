const fs = require('fs');
const path = require('path');

const root = process.cwd();
const intermediate = path.join(root, '.understand-anything', 'intermediate');
const scan = JSON.parse(fs.readFileSync(path.join(intermediate, 'scan-result.json'), 'utf8'));
const batches = JSON.parse(fs.readFileSync(path.join(intermediate, 'batches.json'), 'utf8')).batches;

const moduleDefs = [
  ['module:curriculum', '课程内容', '课程阶段、课程文档、示例代码、测验和可复用输出。', '课程结构'],
  ['module:site', '静态站点', '把 README、ROADMAP、glossary 和课程文件编译为可浏览网站的前端与构建脚本。', '站点'],
  ['module:automation', '自动化脚本', '用于课程审计、目录生成、链接检查和本地脚手架的维护脚本。', '自动化'],
  ['module:project-docs', '项目文档', '面向贡献者和学习者的仓库级说明、规范和治理文件。', '文档'],
  ['module:configuration', '仓库配置', '编辑器、部署、CI、依赖和工具配置文件。', '配置'],
  ['module:assets', '静态资产', '站点、品牌和展示所需的静态资源。', '资产'],
];

function nodeType(file) {
  const p = file.path;
  if (file.fileCategory === 'config') return 'config';
  if (file.fileCategory === 'docs') return 'document';
  if (file.fileCategory === 'infra') {
    if (p.startsWith('.github/workflows/') || p.includes('Jenkinsfile') || p.includes('.gitlab-ci')) return 'pipeline';
    if (p.endsWith('.tf') || p.endsWith('.tfvars') || p.includes('Vagrantfile')) return 'resource';
    return 'service';
  }
  if (file.fileCategory === 'data') {
    if (/\.(graphql|gql|proto|prisma)$/i.test(p)) return 'schema';
    if (/\.(ya?ml|json)$/i.test(p) && /(openapi|swagger)/i.test(p)) return 'endpoint';
    return 'table';
  }
  return 'file';
}

function idFor(file) {
  return `${nodeType(file)}:${file.path}`;
}

function moduleFor(file) {
  const p = file.path;
  if (p.startsWith('phases/')) return 'module:curriculum';
  if (p.startsWith('site/')) return 'module:site';
  if (p.startsWith('scripts/') || p === 'requirements.txt') return 'module:automation';
  if (p.startsWith('assets/')) return 'module:assets';
  if (p.startsWith('.github/') || p.startsWith('.vscode/') || ['vercel.json', '.coderabbit.yaml', '.env'].includes(p)) return 'module:configuration';
  if (/\.(md|rst|txt)$/i.test(p) || ['AGENTS.md', 'README.md', 'ROADMAP.md', 'CONTRIBUTING.md', 'CHANGELOG.md', 'FORKING.md', 'CODE_OF_CONDUCT.md', 'LICENSE', 'SPONSORS.md'].includes(p)) return 'module:project-docs';
  return 'module:configuration';
}

function complexity(lines) {
  if (lines > 200) return 'complex';
  if (lines >= 50) return 'moderate';
  return 'simple';
}

function tagsFor(file) {
  const p = file.path;
  const tags = [];
  if (p.startsWith('phases/')) tags.push('课程');
  if (p.includes('/docs/')) tags.push('lesson-doc');
  if (p.includes('/code/')) tags.push('示例代码');
  if (p.includes('/tests/') || /test|spec/i.test(path.basename(p))) tags.push('test');
  if (p.includes('/outputs/')) tags.push('可复用产物');
  if (file.fileCategory === 'docs') tags.push('documentation');
  if (file.fileCategory === 'config') tags.push('configuration');
  if (file.fileCategory === 'infra') tags.push('infrastructure');
  if (file.fileCategory === 'script') tags.push('automation');
  if (file.fileCategory === 'markup') tags.push('frontend');
  if (file.fileCategory === 'data') tags.push('data');
  tags.push(file.language || 'unknown');
  return [...new Set(tags)].slice(0, 5);
}

function summaryFor(file) {
  const p = file.path;
  if (p === 'README.md') return '仓库入口文档，说明课程定位、学习路径、阶段结构和内置技能。';
  if (p === 'ROADMAP.md') return '课程路线图，记录 20 个阶段的课程规划与完成状态。';
  if (p.startsWith('phases/') && p.includes('/docs/')) return '课程讲义文档，解释该课的目标、概念、实现思路和学习产物。';
  if (p.startsWith('phases/') && p.includes('/quiz.json')) return '课程测验数据，按 pre、check 和 post 阶段组织理解检查问题。';
  if (p.startsWith('phases/') && p.includes('/code/')) return '课程配套实现代码或测试，用于从底层实现并验证该课的核心技术。';
  if (p.startsWith('phases/') && p.includes('/outputs/')) return '课程输出产物，可作为 prompt、skill、agent 或工具配置复用于实际工作流。';
  if (p.startsWith('site/')) return '静态课程网站的一部分，用于渲染目录、课程页、搜索或站点交互。';
  if (p.startsWith('scripts/')) return '仓库维护脚本，用于审计课程结构、生成目录、检查链接或创建课程脚手架。';
  if (p.startsWith('glossary/')) return '术语表文档，维护课程中复用概念的统一定义。';
  if (p.startsWith('.github/')) return 'GitHub 仓库自动化配置，用于 PR 模板、资助信息或 CI 工作流。';
  if (file.fileCategory === 'config') return '项目配置文件，控制仓库工具、部署、编辑器或运行时行为。';
  if (file.fileCategory === 'docs') return '项目文档文件，补充课程、贡献流程或仓库治理信息。';
  return '项目文件，属于课程仓库的内容、工具或站点支持结构。';
}

function makeNode(file) {
  return {
    id: idFor(file),
    type: nodeType(file),
    name: path.basename(file.path),
    filePath: file.path,
    summary: summaryFor(file),
    complexity: complexity(file.sizeLines),
    tags: tagsFor(file),
    language: file.language,
  };
}

function edge(source, target, type, weight = 0.5) {
  return { source, target, type, direction: 'forward', weight };
}

const byPath = new Map(scan.files.map(f => [f.path, f]));
const batchByPath = new Map();
for (const batch of batches) {
  for (const file of batch.files) batchByPath.set(file.path, batch.batchIndex);
}

const moduleNodes = moduleDefs.map(([id, name, summary, tag]) => ({
  id,
  type: 'module',
  name,
  summary,
  complexity: 'moderate',
  tags: [tag, '架构'],
}));
fs.writeFileSync(path.join(intermediate, 'batch-0.json'), JSON.stringify({ nodes: moduleNodes, edges: [] }, null, 2));

for (const batch of batches) {
  const nodes = batch.files.map(makeNode);
  const edges = [];
  for (const file of batch.files) {
    const src = idFor(file);
    edges.push(edge(moduleFor(file), src, 'contains', 1.0));
    for (const targetPath of scan.importMap[file.path] || []) {
      const target = byPath.get(targetPath);
      if (target) edges.push(edge(src, idFor(target), 'imports', 0.7));
    }
  }
  fs.writeFileSync(path.join(intermediate, `batch-${batch.batchIndex}.json`), JSON.stringify({ nodes, edges }, null, 2));
}

const testPairs = [];
for (const file of scan.files) {
  if (!(/\/tests?\//.test(file.path) || /(^|[._-])(test|spec)([._-]|$)/i.test(path.basename(file.path)))) continue;
  const prodPath = file.path
    .replace('/tests/', '/')
    .replace('/test/', '/')
    .replace(/(^|\/)test_/, '$1')
    .replace(/([._-])(test|spec)(?=\.)/i, '');
  if (byPath.has(prodPath)) testPairs.push(edge(idFor(byPath.get(prodPath)), idFor(file), 'tested_by', 0.5));
}
if (testPairs.length) {
  fs.writeFileSync(path.join(intermediate, 'batch-9999.json'), JSON.stringify({ nodes: [], edges: testPairs }, null, 2));
}

console.log(`Generated ${batches.length + 1} batch graph files for ${scan.files.length} files.`);
