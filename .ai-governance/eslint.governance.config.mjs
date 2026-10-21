// 治理门禁专用 ESLint 9 扁平配置（零依赖：不引入任何第三方插件/解析器）
//
// 背景：risk-classify 对「整文件内容」匹配硬词（auth/login/payment…）→ 只要页面
// require 了 utils/auth.js 就会被定级 L2；而 L2 下 quality-gate.sh 会对 JS 改动
// 要求 eslint，此前 CI 既未安装 eslint、仓库也没有可用配置 → 纯 JS 的 L2 改动
// 必然 fail-closed（Batch 2 小程序工程卫生因此无法推进，台账 OODA-20261003-002）。
//
// 设计：只使用 ESLint 核心规则（无需网络与插件安装）
//   - error：可确定的结构性缺陷（重复键/不可达/重复声明/空块/条件赋值）
//   - warn ：complexity(15) / max-depth(5) —— 存量债以报告形式呈现，不阻断；
//           Batch 2 内偿还后再收紧为 error（基线债实测见台账 OODA-20261003-002）
export default [
  {
    files: ["**/*.js", "**/*.cjs", "**/*.mjs"],
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: "commonjs",
      globals: {
        // 小程序宿主
        wx: "readonly",
        Page: "readonly",
        Component: "readonly",
        Behavior: "readonly",
        App: "readonly",
        getApp: "readonly",
        getCurrentPages: "readonly",
        // Node/CommonJS
        require: "readonly",
        module: "writable",
        exports: "writable",
        __dirname: "readonly",
        __filename: "readonly",
        process: "readonly",
        Buffer: "readonly",
        global: "readonly",
        console: "readonly",
        setTimeout: "readonly",
        clearTimeout: "readonly",
        setInterval: "readonly",
        clearInterval: "readonly",
        setImmediate: "readonly",
        TextDecoder: "readonly",
        TextEncoder: "readonly",
        URL: "readonly",
        URLSearchParams: "readonly",
        AbortController: "readonly",
        Blob: "readonly",
        FormData: "readonly",
        // 浏览器宿主（web 侧 .js）
        window: "readonly",
        document: "readonly",
        navigator: "readonly",
        location: "readonly",
        fetch: "readonly",
        XMLHttpRequest: "readonly",
        WebSocket: "readonly",
        // Jest（utils/__tests__/*.test.js）
        describe: "readonly",
        it: "readonly",
        test: "readonly",
        expect: "readonly",
        jest: "readonly",
        beforeEach: "readonly",
        afterEach: "readonly",
        beforeAll: "readonly",
        afterAll: "readonly",
      },
    },
    rules: {
      "no-undef": "error",
      "no-dupe-keys": "error",
      "no-dupe-args": "error",
      "no-unreachable": "error",
      "no-redeclare": "error",
      "no-empty": "error",
      "no-cond-assign": "error",
      "no-obj-calls": "error",
      "no-sparse-arrays": "error",
      "complexity": ["warn", 15],
      "max-depth": ["warn", 5],
      // 存量债：未使用变量量大且无行为风险，关闭以免噪声淹没真实问题
      "no-unused-vars": "off",
    },
  },
  {
    // Service Worker（frontend/public/sw.js）：运行在 WorkerGlobalScope 而非 window，
    // 其宿主全局（caches/clients/Response/Headers/skipWaiting…）无法用上面的浏览器块覆盖。
    // 只对本文件声明，避免把 `self` 泄漏到普通脚本里，掩盖真正的 this 误用。
    files: ["**/public/sw.js", "**/service-worker.js", "**/sw.js"],
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: "script",
      globals: {
        self: "readonly",
        caches: "readonly",
        clients: "readonly",
        skipWaiting: "readonly",
        addEventListener: "readonly",
        removeEventListener: "readonly",
        event: "readonly",
        fetch: "readonly",
        Request: "readonly",
        Response: "readonly",
        Headers: "readonly",
        location: "readonly",
        console: "readonly",
        setTimeout: "readonly",
        clearTimeout: "readonly",
        URL: "readonly",
      },
    },
  },
  {
    // .mjs 是 ESM 语义（`export default` / 顶层 import）。上面的通用块为兼容小程序与
    // 构建脚本收进了 `**/*.mjs`（为了继承那套 Node/宿主 globals），但 sourceType 是
    // commonjs → 触发「import/export 只能出现在 module」的解析错误，连治理配置自身与
    // scripts/*.mjs 都被误报。扁平配置按顺序合并，此处只覆盖 sourceType，
    // globals 仍继承上面的通用块，不重复声明。
    files: ["**/*.mjs"],
    languageOptions: { sourceType: "module" },
  },
];
