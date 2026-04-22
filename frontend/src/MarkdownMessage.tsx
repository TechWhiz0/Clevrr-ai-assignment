import ReactMarkdown, { defaultUrlTransform } from "react-markdown";
import type { Components, UrlTransform } from "react-markdown";
import remarkGfm from "remark-gfm";

type Props = {
  content: string;
};

const urlTransform: UrlTransform = (url) => {
  const u = url.trim();
  if (/^data:image\/(png|jpeg|jpg|gif|webp|svg\+xml);base64,/i.test(u)) {
    return u;
  }
  return defaultUrlTransform(url);
};

const components: Components = {
  table({ children, ...props }) {
    return (
      <div className="md-table-wrap">
        <div className="md-table-label">Table</div>
        <div className="md-table-scroll">
          <table {...props}>{children}</table>
        </div>
      </div>
    );
  },
  img({ src, alt, ...props }) {
    if (!src) return null;
    return (
      <figure className="md-chart-figure">
        <img {...props} src={src} alt={alt ?? "Chart"} className="md-chart-img" loading="lazy" />
        {alt ? <figcaption className="md-chart-caption">{alt}</figcaption> : null}
      </figure>
    );
  },
  h2({ children, ...props }) {
    return (
      <h2 {...props} className="md-h2">
        {children}
      </h2>
    );
  },
  h3({ children, ...props }) {
    return (
      <h3 {...props} className="md-h3">
        {children}
      </h3>
    );
  },
  ul({ children, ...props }) {
    return (
      <ul {...props} className="md-ul">
        {children}
      </ul>
    );
  },
  ol({ children, ...props }) {
    return (
      <ol {...props} className="md-ol">
        {children}
      </ol>
    );
  },
  strong({ children, ...props }) {
    return (
      <strong {...props} className="md-strong">
        {children}
      </strong>
    );
  },
  p({ children, ...props }) {
    return (
      <p {...props} className="md-p">
        {children}
      </p>
    );
  },
};

export function MarkdownMessage({ content }: Props) {
  return (
    <div className="md-root">
      <ReactMarkdown remarkPlugins={[remarkGfm]} urlTransform={urlTransform} components={components}>
        {content}
      </ReactMarkdown>
    </div>
  );
}
