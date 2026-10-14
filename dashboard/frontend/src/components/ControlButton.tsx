import { useEffect, useLayoutEffect, useRef, useState, type CSSProperties, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import type { LiveControl } from '../lib/api'
import { sendControl } from '../lib/pnl'

/*
  Dashboard button -> bot command, with a small CONFIRM box right under the button (no browser popup).
  The box is drawn on top of everything (portal), so tables with scroll do not cut it off.
  Click outside / Esc / CANCEL closes it. Errors show in the same box.
*/
export default function ControlButton({
  control, question, confirmText = 'CONFIRM', className, style, title, children,
}: {
  control: LiveControl
  question: string
  confirmText?: string
  className?: string
  style?: CSSProperties
  title?: string
  children: ReactNode
}) {
  const btn = useRef<HTMLButtonElement>(null)
  const box = useRef<HTMLDivElement>(null)
  const [open, setOpen] = useState(false)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const [pos, setPos] = useState<{ top: number; left: number } | null>(null)

  // place the box under the button (or above it if there is no room below)
  useLayoutEffect(() => {
    if (!open || !btn.current) return
    const r = btn.current.getBoundingClientRect()
    const w = 240
    const h = box.current?.offsetHeight ?? 110
    const left = Math.max(8, Math.min(window.innerWidth - w - 8, r.right - w))
    const below = r.bottom + 6
    const top = below + h > window.innerHeight - 8 ? Math.max(8, r.top - h - 6) : below
    setPos({ top, left })
  }, [open, err])

  useEffect(() => {
    if (!open) return
    const close = () => { setOpen(false); setErr(null) }
    const onDown = (e: MouseEvent) => {
      const t = e.target as Node
      if (box.current?.contains(t) || btn.current?.contains(t)) return
      close()
    }
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') close() }
    document.addEventListener('mousedown', onDown)
    document.addEventListener('keydown', onKey)
    window.addEventListener('scroll', close, true)
    window.addEventListener('resize', close)
    return () => {
      document.removeEventListener('mousedown', onDown)
      document.removeEventListener('keydown', onKey)
      window.removeEventListener('scroll', close, true)
      window.removeEventListener('resize', close)
    }
  }, [open])

  const go = async () => {
    setBusy(true)
    const e = await sendControl(control)
    setBusy(false)
    if (e) setErr(e)
    else { setOpen(false); setErr(null) }
  }

  return (
    <>
      <button
        ref={btn}
        className={className}
        style={style}
        title={title}
        onClick={(e) => { e.stopPropagation(); setErr(null); setOpen((o) => !o) }}
      >
        {children}
      </button>
      {open && createPortal(
        <div
          ref={box}
          role="dialog"
          className="text-[10px]"
          onClick={(e) => e.stopPropagation()}
          style={{
            position: 'fixed', top: pos?.top ?? -9999, left: pos?.left ?? -9999, width: 240, zIndex: 1000,
            background: 'var(--panel)', border: '1px solid var(--cyan)', boxShadow: '0 6px 24px rgba(0,0,0,.55)',
            padding: '8px 10px', color: 'var(--text)', textAlign: 'left',
          }}
        >
          <div style={{ whiteSpace: 'pre-line', lineHeight: 1.45 }}>{question}</div>
          {err && <div className="mt-1" style={{ color: 'var(--red)' }}>Not sent: {err}</div>}
          <div className="mt-2 flex justify-end gap-2">
            <button className="chip !px-2 !py-0.5 text-[9px]" onClick={() => { setOpen(false); setErr(null) }} disabled={busy}>CANCEL</button>
            <button
              className="chip !px-2 !py-0.5 text-[9px]"
              style={{ color: 'var(--cyan)', borderColor: 'var(--cyan)' }}
              onClick={() => void go()}
              disabled={busy}
              autoFocus
            >
              {busy ? 'SENDING...' : confirmText}
            </button>
          </div>
        </div>,
        document.body,
      )}
    </>
  )
}
