const steps = [
  {
    number: 1,
    title: 'Upload research papers',
    description: 'Add PDF documents to your research library.',
  },
  {
    number: 2,
    title: 'Process and explore',
    description: 'We extract text, create chunks, and prepare papers for analysis.',
  },
  {
    number: 3,
    title: 'Generate research summaries',
    description: 'Get plain-language summaries with evidence from your documents.',
  },
  {
    number: 4,
    title: 'Compare papers',
    description: 'Find similarities and differences across multiple research papers.',
  },
  {
    number: 5,
    title: 'Verify claims',
    description: 'Check whether statements in summaries are supported by the paper.',
  },
];

export default function GetStarted() {
  return (
    <section aria-labelledby="get-started-heading" className="rounded-xl border border-gray-200 bg-white shadow-sm overflow-hidden">
      {/* Card header */}
      <div className="px-5 py-4 border-b border-gray-100">
        <h2 id="get-started-heading" className="text-base font-semibold text-gray-900">
          Get Started
        </h2>
      </div>

      {/* Steps */}
      <ol className="divide-y divide-gray-100" aria-label="Getting started steps">
        {steps.map(({ number, title, description }) => (
          <li key={number} className="flex items-start gap-4 px-5 py-4">
            {/* Step number */}
            <span
              className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-blue-600 text-xs font-bold text-white"
              aria-hidden="true"
            >
              {number}
            </span>
            {/* Step content */}
            <div className="min-w-0">
              <p className="text-sm font-semibold text-gray-900">{title}</p>
              <p className="mt-0.5 text-sm text-gray-500">{description}</p>
            </div>
          </li>
        ))}
      </ol>
    </section>
  );
}
